package main

# ========================================
# Do Not store secrets in ENV variables
# ========================================
secrets_env := [
    "passwd", "password", "pass", "secret", "key",
    "access", "api_key", "apikey", "token", "tkn"
]

deny contains msg if {
    input[i].Cmd == "env"
    val := input[i].Value[_]
    secret := secrets_env[_]
    contains(lower(val), secret)
    msg := sprintf("Line %d: Potential secret in ENV variable: '%s'", [i, val])
}

# ========================================
# Do not use 'latest' tag for base images
# ========================================
deny contains msg if {
    input[i].Cmd == "from"
    val := split(input[i].Value[0], ":")
    count(val) > 1
    lower(val[1]) == "latest"
    msg := sprintf("Line %d: Avoid using 'latest' tag for base images", [i])
}

# ========================================
# Avoid curl/wget bashing (piping to shell)
# ========================================
deny contains msg if {
    input[i].Cmd == "run"
    val := concat(" ", input[i].Value)
    regex.match(`(?i)(curl|wget)[^|]*\|.*(sh|bash)`, val)
    msg := sprintf("Line %d: Avoid curl/wget bashing (piping to shell)", [i])
}

# ========================================
# Do not upgrade system packages
# ========================================
upgrade_commands := ["apk upgrade", "apt-get upgrade", "dist-upgrade", "dnf upgrade", "yum upgrade"]

deny contains msg if {
    input[i].Cmd == "run"
    val := concat(" ", input[i].Value)
    cmd := upgrade_commands[_]
    contains(lower(val), lower(cmd))
    msg := sprintf("Line %d: Do not upgrade system packages: '%s'", [i, cmd])
}

# ========================================
# Prefer COPY over ADD
# ========================================
deny contains msg if {
    input[i].Cmd == "add"
    msg := sprintf("Line %d: Use COPY instead of ADD", [i])
}

# ========================================
# Always define a non-root USER
# ========================================
has_user if {
    some i
    input[i].Cmd == "user"
}

deny contains msg if {
    not has_user
    msg := "Specify a non-root USER in the Dockerfile"
}

# ========================================
# Do not run as root
# ========================================
forbidden_users := ["root", "toor", "0"]

deny contains msg if {
    input[i].Cmd == "user"
    val := concat(" ", input[i].Value)
    user := forbidden_users[_]
    contains(lower(val), user)
    msg := sprintf("Line %d: Do not run as root user: '%s'", [i, val])
}

# ========================================
# Do not use sudo
# ========================================
deny contains msg if {
    input[i].Cmd == "run"
    val := concat(" ", input[i].Value)
    contains(lower(val), "sudo")
    msg := sprintf("Line %d: Do not use 'sudo' command", [i])
}