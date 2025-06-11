# Use MathWorks MATLAB R2025a as the base image
FROM mathworks/matlab:r2025a

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    APP_HOME=/app

# Create app home directory early
RUN mkdir -p $APP_HOME

# Install system dependencies from the original Dockerfile, ensuring Python 3.12 is prioritized if available
# The MATLAB image is Ubuntu-based.
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    # System essentials
    ca-certificates \
    procps \
    sudo \
    net-tools \
    dnsutils \
    iproute2 \
    lsof \
    # Python environment (prefer Python 3.12 if available, else system's python3 for venv)
    # The mathworks image might have its own Python. We aim for 3.12 for consistency.
    # Installing python3.12 explicitly might conflict if base has different default python3.
    # For now, assume base's python3 is sufficient or we install pip for it.
    # If python3.12 is a strict requirement and not the default python3, this needs adjustment (e.g. PPA)
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    # Development tools
    build-essential \
    golang \
    nodejs \
    npm \
    # Version control
    git \
    tig \
    # Security & containerization
    firejail \
    apparmor \
    # Text editors
    vim \
    nano \
    neovim \
    # System monitoring & analysis
    htop \
    glances \
    neofetch \
    iotop \
    sysstat \
    strace \
    ltrace \
    # Network utilities
    curl \
    wget \
    nmap \
    traceroute \
    mtr-tiny \
    openssh-client \
    socat \
    netcat-openbsd \
    httpie \
    iperf3 \
    tshark \
    # Browsers and web tools (Chromium might be heavy, consider if needed with MATLAB base)
    # chromium-browser \
    # chromium-driver \
    # File management
    tree \
    rsync \
    ncdu \
    ranger \
    # Compression utilities
    tar \
    gzip \
    pigz \
    bzip2 \
    xz-utils \
    p7zip-full \
    unrar \
    zstd \
    lzip \
    lzma \
    cpio \
    unzip \
    zip \
    # Data formats & processing
    jq \
    yq \
    # Search tools (fd-find might be fd on some systems)
    fd-find \
    ripgrep \
    silversearcher-ag \
    # Shell & terminal utilities
    tmux \
    byobu \
    fzf \
    zsh \
    watch \
    entr \
    moreutils \
    parallel \
    pv \
    less \
    bc \
    bat \
    ncurses-bin \
    # Media processing (FFmpeg might be heavy, consider if needed)
    # ffmpeg \
    # libavcodec-dev \
    # libavformat-dev \
    # libswscale-dev \
    # libavfilter-dev \
    imagemagick \
    libmagickwand-dev \
    # Security and penetration testing tools (subset, consider if all needed)
    # nmap is already listed above
    # nikto \
    # sqlmap \
    # dirb \
    # gobuster \
    # hydra \
    # hashcat \
    # john \
    # Encryption utilities
    gnupg \
    openssl \
    cryptsetup \
    age \
    keychain \
    libssl-dev \
    libffi-dev \
    # Database clients
    postgresql-client \
    mysql-client \
    redis-tools \
    sqlite3 \
    # Infrastructure as code tools
    # ansible \
    hugo \
    # Document processing & language tools
    pandoc \
    tesseract-ocr \
    tesseract-ocr-eng \
    hunspell \
    translate-shell \
    poppler-utils \
    # wkhtmltopdf \ # Often problematic, consider alternatives if needed
    figlet \
    wordnet \
    aspell \
    aspell-en \
    libxml2-dev \
    libxslt1-dev \
    # System dependencies for scipy
    libatlas-base-dev \
    gfortran \
    libblas-dev \
    liblapack-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user 'appuser' and set up its home directory and permissions.
# The MATLAB image might run as root or a 'matlab' user. We create our own for consistency.
RUN useradd -ms /bin/bash -d $APP_HOME appuser && \
    chown -R appuser:appuser $APP_HOME
# Add appuser to sudoers with NOPASSWD for firejail.
RUN echo "appuser ALL=(ALL) NOPASSWD: /usr/bin/firejail" >> /etc/sudoers

# Define working directory (already created, now set as WORKDIR)
WORKDIR $APP_HOME

# Copy application files
COPY --chown=appuser:appuser cmcp $APP_HOME/cmcp
COPY --chown=appuser:appuser tests $APP_HOME/tests
COPY --chown=appuser:appuser scripts $APP_HOME/scripts
COPY --chown=appuser:appuser resources $APP_HOME/resources
COPY --chown=appuser:appuser pyproject.toml $APP_HOME/pyproject.toml
COPY --chown=appuser:appuser pytest.ini $APP_HOME/pytest.ini
COPY --chown=appuser:appuser README.md /app/

# Copy AppArmor profiles (but don't load them during build)
# Ensure the /etc/apparmor.d directory exists
RUN mkdir -p /etc/apparmor.d
COPY apparmor/mcp-bash /etc/apparmor.d/
COPY apparmor/mcp-python /etc/apparmor.d/


# Switch to appuser for venv creation and pip installs
USER appuser

# Setup Python virtual environment using the python3 available
# (Aiming for 3.12, but this uses the default python3 from the image + installed pip/venv)
RUN python3 -m venv $APP_HOME/.venv
ENV PATH="$APP_HOME/.venv/bin:$PATH"

# Install/Upgrade pip, setuptools, wheel, then uv
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir uv

# Install project dependencies using uv
RUN uv pip install --no-cache-dir -e .

# Ensure scripts are executable by appuser
RUN chmod +x $APP_HOME/scripts/*.sh

# USER appuser is already set

# Expose API port (if your app is a server)
EXPOSE 8000

# Start the application using startup script
ENTRYPOINT ["/app/scripts/startup.sh"]
CMD ["--all-in-one"]