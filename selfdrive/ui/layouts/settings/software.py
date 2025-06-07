from openpilot.system.ui.lib.application import Widget
from openpilot.system.ui.lib.list_view import ListView, button_item, text_item


class SoftwareLayout(Widget):
  def __init__(self):
    super().__init__()
    items = [
      text_item("Current Version", ""),
      button_item("Download", "CHECK", callback=self._on_download_update),
      button_item("Install Update", "INSTALL", callback=self._on_install_update),
      button_item("Target Branch", "SELECT", callback=self._on_select_branch),
      button_item("Uninstall", "UNINSTALL", callback=self._on_uninstall),
    ]

    self._list_widget = ListView(items)

  def _render(self, rect):
    self._list_widget.render(rect)

  def _on_download_update(self): pass
  def _on_install_update(self): pass
  def _on_select_branch(self): pass
  def _on_uninstall(self): pass
