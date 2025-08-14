# Use Debian Bullseye as base
FROM debian:bullseye

# Set up a non-root user
RUN groupadd --gid 1000 node && \
    useradd --uid 1000 --gid node --shell /bin/bash --create-home node

# Install dependencies
RUN apt update && \
    apt install -y curl wget gnupg dirmngr xz-utils libatomic1 ca-certificates openjdk-11-jdk bash && \
    rm -rf /var/lib/apt/lists/*

# Install Node.js and npm
ENV NODE_VERSION=20.18.1
RUN ARCH=$(dpkg --print-architecture) && \
    curl -fsSLO "https://nodejs.org/dist/v$NODE_VERSION/node-v$NODE_VERSION-linux-$ARCH.tar.xz" && \
    tar -xJf "node-v$NODE_VERSION-linux-$ARCH.tar.xz" -C /usr/local --strip-components=1 && \
    rm "node-v$NODE_VERSION-linux-$ARCH.tar.xz" && \
    ln -s /usr/local/bin/node /usr/local/bin/nodejs

# Verify Node.js and npm installation
RUN node --version && npm --version

# Install Yarn
ENV YARN_VERSION=1.22.22
RUN curl -fsSLO "https://yarnpkg.com/downloads/$YARN_VERSION/yarn-v$YARN_VERSION.tar.gz" && \
    tar -xzf yarn-v$YARN_VERSION.tar.gz -C /opt/ && \
    ln -s /opt/yarn-v$YARN_VERSION/bin/yarn /usr/local/bin/yarn && \
    ln -s /opt/yarn-v$YARN_VERSION/bin/yarnpkg /usr/local/bin/yarnpkg && \
    rm yarn-v$YARN_VERSION.tar.gz

# Install Firebase CLI globally
RUN npm install -g firebase-tools

# Copy the Firestore emulator script
COPY entrypoint.sh /usr/local/bin/docker-entrypoint.sh

# Ensure the script is executable
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Set working directory
WORKDIR /firebase-emulator

# Copy firebase.json into the container
COPY firebase.json .
COPY firestore.rules .
COPY storage.rules  .

# Entrypoint and default command
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
