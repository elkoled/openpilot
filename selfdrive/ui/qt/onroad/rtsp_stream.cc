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
  args << "-rtsp_transport" << "tcp"
       << "-i" << url
       << "-f" << "mjpeg"
       << "-q:v" << "5"
       << "-r" << "10"
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
  buffer.append(ffmpeg_process->readAllStandardOutput());
  processBuffer();
}

void RtspStream::processBuffer() {
  static const QByteArray jpeg_start = QByteArray::fromHex("FFD8");
  static const QByteArray jpeg_end = QByteArray::fromHex("FFD9");

  while (true) {
    int start = buffer.indexOf(jpeg_start);
    if (start == -1) return;

    int end = buffer.indexOf(jpeg_end, start + 2);
    if (end == -1) return;

    QByteArray frame_data = buffer.mid(start, end - start + 2);
    buffer = buffer.mid(end + 2);

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