#pragma once

#include <QObject>
#include <QPixmap>
#include <QProcess>
#include <QTimer>
#include <functional>

class RtspStream : public QObject {
  Q_OBJECT

public:
  explicit RtspStream(QObject *parent = nullptr);
  ~RtspStream();

  void start(const QString &url);
  void stop();
  bool isActive() const { return active; }
  bool hasFrame() const { return !current_frame.isNull(); }
  const QPixmap &frame() const { return current_frame; }
  void setFrameCallback(std::function<void()> callback) { frame_callback = callback; }

private slots:
  void onReadyRead();
  void onFinished(int exitCode, QProcess::ExitStatus exitStatus);
  void onError(QProcess::ProcessError error);

private:
  void processBuffer();
  void extractFrame(const QByteArray &frame_data);

  QProcess *ffmpeg_process;
  QPixmap current_frame;
  QByteArray buffer;
  bool active;
  std::function<void()> frame_callback;
  QTimer *restart_timer;
};