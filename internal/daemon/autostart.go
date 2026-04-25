package daemon

import (
	"fmt"
	"runtime"
)

// EnableAutostart enables the platform-native user login autostart mechanism.
func EnableAutostart(configPath string, port int) error {
	switch runtime.GOOS {
	case "darwin":
		return enableLaunchdAutostart(configPath, port)
	case "linux":
		return enableSystemdAutostart(configPath, port)
	default:
		return fmt.Errorf("autostart is not supported on %s", runtime.GOOS)
	}
}

// DisableAutostart disables the platform-native user login autostart mechanism.
func DisableAutostart() error {
	switch runtime.GOOS {
	case "darwin":
		return disableLaunchdAutostart()
	case "linux":
		return disableSystemdAutostart()
	default:
		return fmt.Errorf("autostart is not supported on %s", runtime.GOOS)
	}
}

// AutostartStatus reports the platform-native user login autostart status.
func AutostartStatus() error {
	switch runtime.GOOS {
	case "darwin":
		return launchdAutostartStatus()
	case "linux":
		return systemdAutostartStatus()
	default:
		return fmt.Errorf("autostart is not supported on %s", runtime.GOOS)
	}
}
