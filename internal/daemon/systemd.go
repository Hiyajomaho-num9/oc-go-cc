package daemon

import (
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"text/template"
)

const systemdServiceTemplate = `[Unit]
Description=oc-go-cc proxy
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
ExecStart={{quote .BinaryPath}} serve --config {{quote .ConfigPath}}{{if .Port}} --port {{.Port}}{{end}}
Restart=on-failure
RestartSec=3
Environment={{quote .EnvAssignment}}
EnvironmentFile=-{{escape .EnvFile}}

[Install]
WantedBy=default.target
`

// SystemdServiceData holds the values interpolated into the systemd service template.
type SystemdServiceData struct {
	BinaryPath    string
	ConfigPath    string
	Port          int
	EnvAssignment string
	EnvFile       string
}

func enableSystemdAutostart(configPath string, port int) error {
	paths, err := DefaultPaths()
	if err != nil {
		return err
	}
	if err := paths.EnsureConfigDir(); err != nil {
		return err
	}

	if configPath == "" {
		configPath = filepath.Join(paths.ConfigDir, "config.json")
	} else if !filepath.IsAbs(configPath) {
		configPath, err = filepath.Abs(configPath)
		if err != nil {
			return fmt.Errorf("cannot resolve config path: %w", err)
		}
	}

	serviceDir := filepath.Dir(paths.SystemdServicePath)
	if err := os.MkdirAll(serviceDir, 0755); err != nil {
		return fmt.Errorf("cannot create systemd user directory: %w", err)
	}

	envPath := os.Getenv("PATH")
	if envPath == "" {
		envPath = "/usr/local/bin:/usr/bin:/bin"
	}

	data := SystemdServiceData{
		BinaryPath:    paths.BinaryPath,
		ConfigPath:    configPath,
		Port:          port,
		EnvAssignment: "PATH=" + envPath,
		EnvFile:       paths.EnvFile,
	}

	tmpl, err := template.New("systemd-service").Funcs(template.FuncMap{
		"escape": systemdEscape,
		"quote":  systemdQuote,
	}).Parse(systemdServiceTemplate)
	if err != nil {
		return fmt.Errorf("cannot parse systemd service template: %w", err)
	}

	var buf bytes.Buffer
	if err := tmpl.Execute(&buf, data); err != nil {
		return fmt.Errorf("cannot render systemd service: %w", err)
	}
	if err := os.WriteFile(paths.SystemdServicePath, buf.Bytes(), 0644); err != nil {
		return fmt.Errorf("cannot write systemd service: %w", err)
	}

	if err := runSystemctlUser("daemon-reload"); err != nil {
		return err
	}
	if err := runSystemctlUser("enable", SystemdService); err != nil {
		return err
	}
	if err := runSystemctlUser("restart", SystemdService); err != nil {
		return err
	}

	fmt.Printf("Autostart enabled. %s will start on login.\n", AppName)
	fmt.Printf("  Service: %s\n", paths.SystemdServicePath)
	fmt.Println("Service enabled and started successfully.")
	fmt.Printf("View logs with: journalctl --user -u %s -f\n", SystemdService)
	fmt.Println("To start at boot before interactive login, run: loginctl enable-linger \"$USER\"")
	return nil
}

func disableSystemdAutostart() error {
	paths, err := DefaultPaths()
	if err != nil {
		return err
	}

	serviceExists := true
	if _, err := os.Stat(paths.SystemdServicePath); os.IsNotExist(err) {
		serviceExists = false
	}

	if serviceExists {
		if err := runSystemctlUser("disable", "--now", SystemdService); err != nil {
			fmt.Fprintf(os.Stderr, "note: could not disable systemd user service: %v\n", err)
		}
		if err := os.Remove(paths.SystemdServicePath); err != nil {
			return fmt.Errorf("cannot remove systemd service: %w", err)
		}
		if err := runSystemctlUser("daemon-reload"); err != nil {
			return err
		}
		fmt.Println("Autostart disabled. Systemd user service removed.")
		return nil
	}

	fmt.Println("Autostart is not enabled (no systemd user service found)")
	return nil
}

func systemdAutostartStatus() error {
	paths, err := DefaultPaths()
	if err != nil {
		return err
	}

	serviceExists := true
	if _, err := os.Stat(paths.SystemdServicePath); os.IsNotExist(err) {
		serviceExists = false
	}

	enabledState, enabledErr := systemctlUserOutput("is-enabled", SystemdService)
	activeState, activeErr := systemctlUserOutput("is-active", SystemdService)
	if !serviceExists && enabledErr != nil {
		fmt.Println("Autostart: disabled (no systemd user service found)")
		return nil
	}

	if enabledErr != nil {
		enabledState = "unknown"
	}
	if activeErr != nil {
		activeState = "inactive"
	}

	fmt.Printf("Autostart: %s (systemd user service %s)\n", enabledState, activeState)
	fmt.Printf("  Service: %s\n", paths.SystemdServicePath)
	fmt.Printf("  Logs: journalctl --user -u %s -f\n", SystemdService)
	return nil
}

func runSystemctlUser(args ...string) error {
	output, err := systemctlUserOutput(args...)
	if err != nil {
		return fmt.Errorf("systemctl --user %s failed: %w%s", strings.Join(args, " "), err, formatCommandOutput(output))
	}
	return nil
}

func systemctlUserOutput(args ...string) (string, error) {
	if _, err := exec.LookPath("systemctl"); err != nil {
		return "", fmt.Errorf("systemctl not found: %w", err)
	}

	cmdArgs := append([]string{"--user"}, args...)
	cmd := exec.Command("systemctl", cmdArgs...)
	output, err := cmd.CombinedOutput()
	return strings.TrimSpace(string(output)), err
}

func formatCommandOutput(output string) string {
	if output == "" {
		return ""
	}
	return ": " + output
}

func systemdQuote(value string) string {
	value = systemdEscape(value)
	value = strings.ReplaceAll(value, `\`, `\\`)
	value = strings.ReplaceAll(value, `"`, `\"`)
	return `"` + value + `"`
}

func systemdEscape(value string) string {
	return strings.ReplaceAll(value, "%", "%%")
}
