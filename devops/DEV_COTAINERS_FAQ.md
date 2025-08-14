# Dev Containers FAQ

## What Are Dev Containers?

Dev Containers enable a consistent and reproducible development environment using Docker containerization. They contain the necessary runtimes and dependencies for your project while allowing you to work seamlessly with your local tools.

Read more about Dev Containers: [Ultimate Guide to Dev Containers](https://www.daytona.io/dotfiles/ultimate-guide-to-dev-containers)

---

## Frequently Asked Questions (FAQ)

### Do I need Docker installed?
Yes, Dev Containers utilize Docker containerization technology. You will need [Docker Community Edition (CE)](https://www.docker.com/products/docker-desktop/) or higher installed.

### Do Dev Containers replace my local development environment?
No, Dev Containers run within your local environment and development tools. They simply provide a consistent runtime and dependency setup for your project.

### Can I commit and push from within a Dev Container?
Yes, Dev Containers mount your local source code into the container, allowing you to commit, push, pull, and work with Git as usual.

### Do I need an internet connection to use Dev Containers?
- **First-time setup:** Yes, an internet connection is required to pull dependencies when a Dev Container is built for the first time.
- **Subsequent usage:** No, once built, Dev Containers do not require an internet connection. However, if your project includes commands like `npm install`, an internet connection will be needed for package installations.

### Are Dev Containers platform agnostic?
Yes, Dev Containers can be used on **Windows, macOS, and Linux** since they rely on Docker containerization. The experience may slightly differ between operating systems, but the functionality remains the same.

### Can I use any code editor/IDE with Dev Containers?
Dev Containers work with any editor that supports the **Remote - Containers** extension, including:
- [VS Code](https://code.visualstudio.com/)
- Visual Studio
- Atom
- Sublime Text  
- And more...

---

## Get Started
1. Install [Docker](https://www.docker.com/products/docker-desktop/)
2. Install an editor that supports **Dev Containers** (e.g., VS Code)
3. Open your project in a Dev Container and start coding!

For more details, visit the [official guide](https://www.daytona.io/dotfiles/ultimate-guide-to-dev-containers).
