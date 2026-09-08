def retryWithDelay(int maxRetries, int delay, Closure body) {
  for (int i = 0; i < maxRetries; i++) {
    try {
      return body()
    } catch (Exception e) {
      sleep(delay)
    }
  }
  throw Exception("Failed after ${maxRetries} retries")
}

def device(String ip, String step_label, String cmd) {
  withCredentials([file(credentialsId: 'id_rsa', variable: 'key_file')]) {
    def ssh_cmd = """
ssh -o ControlMaster=auto -o ControlPath=/tmp/ssh_control_%C -o ControlPersist=yes -o ConnectTimeout=5 -o ServerAliveInterval=5 -o ServerAliveCountMax=2 -o BatchMode=yes -o StrictHostKeyChecking=no -i ${key_file} 'comma@${ip}' exec /usr/bin/bash <<'END'

set -e

export TERM=xterm-256color

shopt -s huponexit # kill all child processes when the shell exits

export CI=1
export PYTHONWARNINGS=error
export COMMA_CACHE=/data/tmp/comma_download_cache
#export LOGPRINT=debug # this has gotten too spammy...
export TEST_DIR=${env.TEST_DIR}
export SOURCE_DIR=${env.SOURCE_DIR}
export GIT_BRANCH=${env.GIT_BRANCH}
export GIT_COMMIT=${env.GIT_COMMIT}
export CI_ARTIFACTS_TOKEN=${env.CI_ARTIFACTS_TOKEN}
export GITHUB_COMMENTS_TOKEN=${env.GITHUB_COMMENTS_TOKEN}
export AZURE_TOKEN='${env.AZURE_TOKEN}'
# only use 1 thread since most require real hardware that can't be shared
export PYTEST_ADDOPTS="-n0 -s"


export GIT_SSH_COMMAND="ssh -i /data/gitkey"

source ~/.bash_profile
if [ -f /AGNOS ]; then
  source /etc/profile

  rm -rf /tmp/tmp*
  rm -rf ~/.commacache
  rm -rf /dev/shm/*
  rm -rf /dev/tmp/tmp*

  if ! systemctl is-active --quiet systemd-resolved; then
    echo "restarting resolved"
    sudo systemctl start systemd-resolved
    sleep 3
  fi

  # restart aux USB
  if [ -e /sys/bus/usb/drivers/hub/3-0:1.0 ]; then
    echo "restarting aux usb"
    echo "3-0:1.0" | sudo tee /sys/bus/usb/drivers/hub/unbind
    sleep 0.5
    echo "3-0:1.0" | sudo tee /sys/bus/usb/drivers/hub/bind
  fi
fi
if [ -f /data/openpilot/launch_env.sh ]; then
  source /data/openpilot/launch_env.sh
fi

export LD_LIBRARY_PATH="\$(python -c 'import ffmpeg; print(ffmpeg.LIB_DIR)'):/usr/local/lib:\${LD_LIBRARY_PATH:-}"

ln -snf ${env.TEST_DIR} /data/pythonpath

cd ${env.TEST_DIR} || true
time ${cmd}
END"""

    sh script: ssh_cmd, label: step_label
  }
}

def deviceStage(String stageName, String deviceType, List extra_env, def steps) {
  stage(stageName) {
    if (currentBuild.result != null) {
        return
    }

    if (isReplay()) {
      error("REPLAYING TESTS IS NOT ALLOWED. FIX THEM INSTEAD.")
    }

    def extra = extra_env.collect { "export ${it}" }.join('\n');
    def branch = env.BRANCH_NAME ?: 'master';
    def gitDiff = sh returnStdout: true, script: 'set +x\ncurl -s -H "Authorization: Bearer ${GITHUB_COMMENTS_TOKEN}" https://api.github.com/repos/commaai/openpilot/compare/master...${GIT_BRANCH} | jq .files[].filename || echo "/"', label: 'Getting changes'

    lock(resource: "", label: deviceType, inversePrecedence: true, variable: 'device_ip', quantity: 1, resourceSelectStrategy: 'random') {
      docker.image('ghcr.io/commaai/alpine-ssh').inside('--user=root') {
        timeout(time: 35, unit: 'MINUTES') {
          retry (3) {
            def date = sh(script: 'date', returnStdout: true).trim();
            device(device_ip, "set time", "date -s '" + date + "'")
            device(device_ip, "git checkout", extra + "\n" + readFile("openpilot/selfdrive/test/setup_device_ci.sh"))
          }
          steps.each { item ->
            def name = item[0]
            def cmd = item[1]

            def args = item[2]
            def diffPaths = args.diffPaths ?: []
            def cmdTimeout = args.timeout ?: 9999

            if (branch != "master" && !branch.contains("__jenkins_loop_") && diffPaths && !hasPathChanged(gitDiff, diffPaths)) {
              println "Skipping ${name}: no changes in ${diffPaths}."
              return
            } else {
              timeout(time: cmdTimeout, unit: 'SECONDS') {
                device(device_ip, name, cmd)
              }
            }
          }
        }
      }
    }
  }
}

def hasPathChanged(String gitDiff, List<String> paths) {
  for (path in paths) {
    if (gitDiff.contains(path)) {
      return true
    }
  }
  return false
}

def isReplay() {
  def replayClass = "org.jenkinsci.plugins.workflow.cps.replay.ReplayCause"
  return currentBuild.rawBuild.getCauses().any{ cause -> cause.toString().contains(replayClass) }
}

def setupCredentials() {
  withCredentials([
    string(credentialsId: 'azure_token', variable: 'AZURE_TOKEN'),
  ]) {
    env.AZURE_TOKEN = "${AZURE_TOKEN}"
  }

  withCredentials([
    string(credentialsId: 'ci_artifacts_pat', variable: 'CI_ARTIFACTS_TOKEN'),
  ]) {
    env.CI_ARTIFACTS_TOKEN = "${CI_ARTIFACTS_TOKEN}"
  }

  withCredentials([
    string(credentialsId: 'post_comments_github_pat', variable: 'GITHUB_COMMENTS_TOKEN'),
  ]) {
    env.GITHUB_COMMENTS_TOKEN = "${GITHUB_COMMENTS_TOKEN}"
  }
}

def step(String name, String cmd, Map args = [:]) {
  return [name, cmd, args]
}

// Development job: exercise the upstream Chestnut stage on the dedicated bench.
node {
  env.CI = "1"
  env.PYTHONWARNINGS = "error"
  env.TEST_DIR = "/data/chestnut-ci-workspace/test"
  env.SOURCE_DIR = "/data/chestnut-ci-workspace/source/"
  setupCredentials()
  def revision = checkout(scm)
  env.GIT_BRANCH = revision.GIT_BRANCH
  env.GIT_COMMIT = revision.GIT_COMMIT
  properties([disableConcurrentBuilds()])
        deviceStage("chestnut", "mici-chestnut-ci", ["UNSAFE=1", "CHESTNUT=1"], [
          step("compile big model", "./openpilot/selfdrive/test/compile_chestnut.sh"),
        ])

}
