// Runs on the existing Jenkins server; the bench is an SSH target, not an agent.
def run(String commit, String repository) {
  if (!(commit ==~ /[0-9a-f]{40}/)) {
    error('Expected an exact Git commit')
  }
  if (!(repository in ['https://github.com/elkoled/openpilot', 'https://github.com/elkoled/openpilot.git',
                       'https://github.com/commaai/openpilot', 'https://github.com/commaai/openpilot.git'])) {
    error('Chestnut CI only supports the commaai repository and the elkoled test fork')
  }
  stage('Chestnut compile + inference') {
    timeout(time: 65, unit: 'MINUTES') {
      lock(label: 'mici-chestnut-ci', quantity: 1, variable: 'CHESTNUT_HOST') {
        docker.image('ghcr.io/commaai/alpine-ssh').inside('--user=root') {
          withCredentials([file(credentialsId: 'id_rsa', variable: 'key_file')]) {
            withEnv(["CHESTNUT_COMMIT=${commit}", "CHESTNUT_REPOSITORY=${repository}"]) {
              try {
                sh '''#!/bin/sh
set -eu
case "$CHESTNUT_HOST" in comma-de2e7866) ;; *) echo "Unexpected CI bench: $CHESTNUT_HOST"; exit 1 ;; esac
ssh -o BatchMode=yes -o ConnectTimeout=5 -o ServerAliveInterval=5 -o ServerAliveCountMax=2 \
  -o StrictHostKeyChecking=accept-new -i "$key_file" "comma@$CHESTNUT_HOST" \
  "timeout --signal=TERM --kill-after=30s 3600 bash -s -- '$CHESTNUT_HOST' '$CHESTNUT_COMMIT' '$CHESTNUT_REPOSITORY'" \
  < tools/ci/chestnut/remote.sh
'''
              } finally {
                // Stop any surviving remote process group before releasing the lock.
                sh '''#!/bin/sh
set -eu
ssh -o BatchMode=yes -o ConnectTimeout=5 -o ServerAliveInterval=5 -o ServerAliveCountMax=2 \
  -o StrictHostKeyChecking=accept-new -i "$key_file" "comma@$CHESTNUT_HOST" \
  "bash -s -- '$CHESTNUT_COMMIT'" < tools/ci/chestnut/cancel.sh
'''
                // Collect small reports only; never copy model weights into Jenkins.
                sh '''#!/bin/sh
mkdir -p chestnut-results
for extension in xml log json txt; do
  scp -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new -i "$key_file" \
    "comma@$CHESTNUT_HOST:/data/chestnut-ci-workspace/reports/$CHESTNUT_COMMIT/*.$extension" chestnut-results/ || true
done
'''
                junit allowEmptyResults: true, testResults: 'chestnut-results/*.xml'
                archiveArtifacts allowEmptyArchive: true, artifacts: 'chestnut-results/*'
              }
            }
          }
        }
      }
    }
  }
}

return this
