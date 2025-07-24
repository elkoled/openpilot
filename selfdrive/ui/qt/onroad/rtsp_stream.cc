#include "selfdrive/ui/qt/onroad/rtsp_stream.h"
#include <QDebug>

RtspStream::RtspStream(QObject *parent)
  : QObject(parent), ffmpeg_process(new QProcess(this)),
    active(false), frame_callback(nullptr), restart_timer(new QTimer(this)) {
  restart_timer->setSingleShot(true);
  connect(restart_timer, &QTimer::timeout, this, [this]() {
    if (active) {
      QString url = ffmpeg_process->property("rtsp_url").toString();
      start(url);
    }
  });
}

RtspStream::~RtspStream() {
  stop();
}

void RtspStream::start(const QString &url) {
  if (active) stop();

  active = true;
  buffer.clear();
  current_frame = QPixmap();

  ffmpeg_process->setProperty("rtsp_url", url);

  QStringList args;
  args << "-fflags" << "nobuffer+flush_packets"
      << "-flags" << "low_delay"
      << "-rtsp_transport" << "udp"
      << "-probesize" << "32"
      << "-analyzeduration" << "0"
      << "-max_delay" << "0"
      << "-i" << url
      << "-vf" << "hflip"
      << "-f" << "mjpeg"
      << "-q:v" << "8"
      << "-r" << "30"
      << "-tune" << "zerolatency"
      << "-";

  connect(ffmpeg_process, &QProcess::readyReadStandardOutput, this, &RtspStream::onReadyRead);
  connect(ffmpeg_process, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
          this, &RtspStream::onFinished);
  connect(ffmpeg_process, &QProcess::errorOccurred, this, &RtspStream::onError);

  ffmpeg_process->start("ffmpeg", args);
}

void RtspStream::stop() {
  if (!active) return;

  active = false;
  restart_timer->stop();

  if (ffmpeg_process && ffmpeg_process->state() != QProcess::NotRunning) {
    ffmpeg_process->kill();
    if (!ffmpeg_process->waitForFinished(3000)) {
      ffmpeg_process->terminate();
    }
  }

  ffmpeg_process->disconnect();
  buffer.clear();
}

void RtspStream::onReadyRead() {
  if (!active) return;
  QByteArray newData = ffmpeg_process->readAllStandardOutput();
  buffer.append(newData);
  processBuffer();
}

void RtspStream::processBuffer() {
  static const QByteArray jpeg_start = QByteArray::fromHex("FFD8");
  static const QByteArray jpeg_end = QByteArray::fromHex("FFD9");

  const int MAX_BUFFER_SIZE = 65536;
  const int FRAME_SEARCH_LIMIT = 32768;

  if (buffer.size() > MAX_BUFFER_SIZE) {
    buffer = buffer.right(FRAME_SEARCH_LIMIT);
  }

  while (buffer.size() > 1024) {
    int start = buffer.indexOf(jpeg_start);
    if (start == -1) {
      if (buffer.size() > 4096) {
        buffer.clear();
      }
      return;
    }

    int end = buffer.indexOf(jpeg_end, start + 2);
    if (end == -1) {
      if (buffer.size() > FRAME_SEARCH_LIMIT) {
        buffer = buffer.right(16384);
      }
      return;
    }

    QByteArray frame_data = buffer.mid(start, end - start + 2);
    buffer.remove(0, end + 2);

    extractFrame(frame_data);
  }
}

void RtspStream::extractFrame(const QByteArray &frame_data) {
  if (!active || frame_data.isEmpty()) return;

  QPixmap new_frame;
  if (new_frame.loadFromData(frame_data, "JPEG")) {
    current_frame = new_frame;
    if (frame_callback) frame_callback();
  }
}

void RtspStream::onFinished(int exitCode, QProcess::ExitStatus exitStatus) {
  if (active) {
    restart_timer->start(1000);
  }
}

void RtspStream::onError(QProcess::ProcessError error) {
  qWarning() << "[RtspStream] Process error:" << error;
}