#!/usr/bin/env python3

import json
from pathlib import Path

import Cocoa
import objc
from Cocoa import NSAlert, NSApplication, NSButton, NSMakeRect, NSWindow
from Foundation import NSObject, NSPoint, NSRect, NSSize

################################################################################
# Adjustable placeholders for your Kolada MCP paths:

KOLADA_UV_PATH = "/Applications/KoladaMCP.app/Contents/Resources/kolada_env/bin/uv"
KOLADA_SRC_PATH = "/Applications/KoladaMCP.app/Contents/Resources/kolada_env/src"
KOLADA_MCP_NAME = "Kolada"

################################################################################
# Claude config helper functions


def get_claude_config_path() -> Path:
    """Return the path to Claude’s config file (adjust if needed)."""
    return (
        Path.home()
        / "Library"
        / "Application Support"
        / "Claude"
        / "claude_desktop_config.json"
    )


def load_claude_config() -> dict:
    """Load the Claude config, return {} if missing or invalid."""
    cfg_path = get_claude_config_path()
    if not cfg_path.exists():
        return {}
    try:
        with cfg_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def save_claude_config(config: dict) -> None:
    """Save the config back to disk, creating the directory if needed."""
    cfg_path = get_claude_config_path()
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with cfg_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


################################################################################
# Core Kolada config logic


def install_kolada():
    """Add (or update) Kolada MCP entry in the Claude config."""
    config = load_claude_config()
    if "mcpServers" not in config:
        config["mcpServers"] = {}

    config["mcpServers"][KOLADA_MCP_NAME] = {
        "command": KOLADA_UV_PATH,
        "args": ["--directory", KOLADA_SRC_PATH, "run", "server.py"],
    }
    save_claude_config(config)


def uninstall_kolada():
    """Remove Kolada from the Claude config if present."""
    config = load_claude_config()
    mcp_servers = config.get("mcpServers", {})
    if KOLADA_MCP_NAME in mcp_servers:
        del mcp_servers[KOLADA_MCP_NAME]
        save_claude_config(config)


def update_kolada():
    """Placeholder for future 'update' logic."""
    pass


################################################################################
# PyObjC GUI classes


class KoladaAppDelegate(NSObject):
    """
    App delegate that sets up a basic window with three buttons:
    'Install/Update', 'Uninstall', and 'Update MCP'.
    """

    window = None

    def applicationDidFinishLaunching_(self, notification):
        # Create a window (nibless) at position (100, 100) with size (400 x 200).
        # StyleMask=15 => titled window, closable, miniaturizable, resizable.
        rect = NSRect(NSPoint(100, 100), NSSize(400, 200))
        style_mask = (
            Cocoa.NSTitledWindowMask
            | Cocoa.NSClosableWindowMask
            | Cocoa.NSResizableWindowMask
            | Cocoa.NSMiniaturizableWindowMask
        )
        self.window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            rect, style_mask, Cocoa.NSBackingStoreBuffered, False
        )
        self.window.setTitle_("Kolada MCP Manager")
        self.window.makeKeyAndOrderFront_(None)

        # Add "Install/Update" button
        install_button = NSButton.alloc().initWithFrame_(NSMakeRect(30, 110, 150, 30))
        install_button.setTitle_("Install/Update Kolada")
        install_button.setBezelStyle_(Cocoa.NSRoundedBezelStyle)
        install_button.setTarget_(self)
        install_button.setAction_(objc.selector(self.onInstall_, signature=b"v@:@"))
        self.window.contentView().addSubview_(install_button)

        # Add "Uninstall" button
        uninstall_button = NSButton.alloc().initWithFrame_(NSMakeRect(30, 70, 150, 30))
        uninstall_button.setTitle_("Uninstall Kolada")
        uninstall_button.setBezelStyle_(Cocoa.NSRoundedBezelStyle)
        uninstall_button.setTarget_(self)
        uninstall_button.setAction_(objc.selector(self.onUninstall_, signature=b"v@:@"))
        self.window.contentView().addSubview_(uninstall_button)

        # Add "Update" button
        update_button = NSButton.alloc().initWithFrame_(NSMakeRect(30, 30, 150, 30))
        update_button.setTitle_("Update Kolada MCP")
        update_button.setBezelStyle_(Cocoa.NSRoundedBezelStyle)
        update_button.setTarget_(self)
        update_button.setAction_(objc.selector(self.onUpdate_, signature=b"v@:@"))
        self.window.contentView().addSubview_(update_button)

    def showAlert(self, title, text, style=Cocoa.NSAlertStyleInformational):
        """
        Helper to show an NSAlert with the given title and message.
        """
        alert = NSAlert.alloc().init()
        alert.setAlertStyle_(style)
        alert.setMessageText_(title)
        alert.setInformativeText_(text)
        alert.runModal()

    def onInstall_(self, sender):
        install_kolada()
        self.showAlert(
            "Kolada MCP", "Kolada MCP has been installed or updated in Claude config."
        )

    def onUninstall_(self, sender):
        uninstall_kolada()
        self.showAlert("Kolada MCP", "Kolada MCP has been removed from Claude config.")

    def onUpdate_(self, sender):
        update_kolada()
        self.showAlert("Kolada MCP", "Update not yet implemented!")

    def applicationWillTerminate_(self, notification):
        pass


################################################################################
# Main entry point


def main():
    app = NSApplication.sharedApplication()
    delegate = KoladaAppDelegate.alloc().init()
    app.setDelegate_(delegate)
    Cocoa.NSApp.run()


if __name__ == "__main__":
    main()
