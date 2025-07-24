#include "selfdrive/ui/qt/onroad/alerts.h"

#include <QPainter>
#include <QDebug>
#include <map>

#include "selfdrive/ui/qt/util.h"

void OnroadAlerts::updateState(const UIState &s) {
  Alert a = getAlert(*(s.sm), s.scene.started_frame);
  bool new_alert = !alert.equal(a);
  if (new_alert) {
    alert = a;

    // rear camera logic
    bool should_show_rear = (alert.text1 == "Reverse\nGear");

    if (should_show_rear && !rear_cam_running) {
      // Start rear camera
      qWarning() << "Starting rear camera...";
      rear_cam.start("rtsp://192.168.1.254");
      rear_cam_running = true;
    } else if (!should_show_rear && rear_cam_running) {
      // Stop rear camera immediately when not needed
      qWarning() << "Stopping rear camera...";
      rear_cam.stop();
      rear_cam_running = false;
    }

    update();
  }
}

void OnroadAlerts::clear() {
  // clean up rear camera when clearing alerts
  if (rear_cam_running) {
    rear_cam.stop();
    rear_cam_running = false;
  }
  alert = {};
  update();
}

OnroadAlerts::Alert OnroadAlerts::getAlert(const SubMaster &sm, uint64_t started_frame) {
  const cereal::SelfdriveState::Reader &ss = sm["selfdriveState"].getSelfdriveState();
  const uint64_t selfdrive_frame = sm.rcv_frame("selfdriveState");

  Alert a = {};
  if (selfdrive_frame >= started_frame) {  // Don't get old alert.
    a = {ss.getAlertText1().cStr(), ss.getAlertText2().cStr(),
         ss.getAlertType().cStr(), ss.getAlertSize(), ss.getAlertStatus()};
  }

  if (!sm.updated("selfdriveState") && (sm.frame - started_frame) > 5 * UI_FREQ) {
    const int SELFDRIVE_STATE_TIMEOUT = 5;
    const int ss_missing = (nanos_since_boot() - sm.rcv_time("selfdriveState")) / 1e9;

    // Handle selfdrive timeout
    if (selfdrive_frame < started_frame) {
      // car is started, but selfdriveState hasn't been seen at all
      a = {tr("sunnypilot Unavailable"), tr("Waiting to start"),
           "selfdriveWaiting", cereal::SelfdriveState::AlertSize::MID,
           cereal::SelfdriveState::AlertStatus::NORMAL};
    } else if (ss_missing > SELFDRIVE_STATE_TIMEOUT && !Hardware::PC()) {
      // car is started, but selfdrive is lagging or died
      if (ss.getEnabled() && (ss_missing - SELFDRIVE_STATE_TIMEOUT) < 10) {
        a = {tr("TAKE CONTROL IMMEDIATELY"), tr("System Unresponsive"),
             "selfdriveUnresponsive", cereal::SelfdriveState::AlertSize::FULL,
             cereal::SelfdriveState::AlertStatus::CRITICAL};
      } else {
        a = {tr("System Unresponsive"), tr("Reboot Device"),
             "selfdriveUnresponsivePermanent", cereal::SelfdriveState::AlertSize::MID,
             cereal::SelfdriveState::AlertStatus::NORMAL};
      }
    }
  }
  return a;
}

void OnroadAlerts::paintEvent(QPaintEvent *event) {
  if (alert.size == cereal::SelfdriveState::AlertSize::NONE) {
    return;
  }
  static std::map<cereal::SelfdriveState::AlertSize, const int> alert_heights = {
    {cereal::SelfdriveState::AlertSize::SMALL, 271},
    {cereal::SelfdriveState::AlertSize::MID, 420},
    {cereal::SelfdriveState::AlertSize::FULL, height()},
  };
  int h = alert_heights[alert.size];

  int margin = 40;
  int radius = 30;
  if (alert.size == cereal::SelfdriveState::AlertSize::FULL) {
    margin = 0;
    radius = 0;
  }
  QRect r = QRect(0 + margin, height() - h + margin, width() - margin*2, h - margin*2);

  QPainter p(this);

  // draw background + gradient
  p.setPen(Qt::NoPen);
  p.setCompositionMode(QPainter::CompositionMode_SourceOver);
  p.setBrush(QBrush(alert_colors[alert.status]));
  p.drawRoundedRect(r, radius, radius);

  QLinearGradient g(0, r.y(), 0, r.bottom());
  g.setColorAt(0, QColor::fromRgbF(0, 0, 0, 0.05));
  g.setColorAt(1, QColor::fromRgbF(0, 0, 0, 0.35));

  p.setCompositionMode(QPainter::CompositionMode_DestinationOver);
  p.setBrush(QBrush(g));
  p.drawRoundedRect(r, radius, radius);
  p.setCompositionMode(QPainter::CompositionMode_SourceOver);

  if (alert.text1 == "Reverse\nGear" && rear_cam_running) {
    if (rear_cam.hasFrame()) {
      QPixmap frame = rear_cam.frame();
      if (!frame.isNull()) {
        QPixmap scaled_frame = frame.scaled(r.size(), Qt::KeepAspectRatioByExpanding, Qt::SmoothTransformation);
        int crop_top = scaled_frame.height() > r.height() ? (scaled_frame.height() - r.height()) / 2 : 0;
        p.drawPixmap(r.x(), r.y(), scaled_frame, 0, crop_top, r.width(), r.height());
      }
    } else {
      p.setFont(InterFont(64));
      p.setPen(QColor(255, 255, 255));
      p.drawText(r, Qt::AlignCenter, "Loading rear camera...");
    }
  } else {
    // text
    const QPoint c = r.center();
    p.setPen(QColor(0xff, 0xff, 0xff));
    p.setRenderHint(QPainter::TextAntialiasing);
    if (alert.size == cereal::SelfdriveState::AlertSize::SMALL) {
      p.setFont(InterFont(74, QFont::DemiBold));
      p.drawText(r, Qt::AlignCenter, alert.text1);
    } else if (alert.size == cereal::SelfdriveState::AlertSize::MID) {
      p.setFont(InterFont(88, QFont::Bold));
      p.drawText(QRect(0, c.y() - 125, width(), 150), Qt::AlignHCenter | Qt::AlignTop, alert.text1);
      p.setFont(InterFont(66));
      p.drawText(QRect(0, c.y() + 21, width(), 90), Qt::AlignHCenter, alert.text2);
    } else if (alert.size == cereal::SelfdriveState::AlertSize::FULL) {
      bool l = alert.text1.length() > 15;
      p.setFont(InterFont(l ? 132 : 177, QFont::Bold));
      p.drawText(QRect(0, r.y() + (l ? 240 : 270), width(), 600), Qt::AlignHCenter | Qt::TextWordWrap, alert.text1);
      p.setFont(InterFont(88));
      p.drawText(QRect(0, r.height() - (l ? 361 : 420), width(), 300), Qt::AlignHCenter | Qt::TextWordWrap, alert.text2);
    }
  }
}