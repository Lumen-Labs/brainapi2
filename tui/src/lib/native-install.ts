import type { ServiceName } from "../constants.js";
import type { PlatformCapabilities, LinuxPackageManager } from "./platform.js";

export interface NativeRecipe {
  install: string[];
  start: string[];
  postInstall?: string[];
  docsUrl?: string;
  unsupported?: string;
}

function brewRecipe(service: ServiceName): NativeRecipe | undefined {
  switch (service) {
    case "redis":
      return {
        install: ["brew install redis"],
        start: ["brew services start redis"],
        docsUrl: "https://redis.io/docs/install/install-redis/install-redis-on-mac-os/",
      };
    case "postgresql":
      return {
        install: ["brew install postgresql@16", "brew install pgvector"],
        start: ["brew services start postgresql@16"],
        postInstall: [
          "createdb \"$USER\" 2>/dev/null || true",
          "psql -d postgres -c \"ALTER USER \\\"$USER\\\" WITH SUPERUSER;\"",
          "echo \"Set POSTGRES_USERNAME / POSTGRES_PASSWORD in .env to match this role.\"",
        ],
        docsUrl: "https://github.com/pgvector/pgvector#installation",
      };
    case "neo4j":
      return {
        install: ["brew install neo4j"],
        start: ["brew services start neo4j"],
        postInstall: [
          "# Default credentials are neo4j/neo4j — you'll be prompted to set a new password",
          "# at first login: http://localhost:7474",
        ],
        docsUrl: "https://neo4j.com/docs/operations-manual/current/installation/osx/",
      };
    case "mongo":
      return {
        install: [
          "brew tap mongodb/brew",
          "brew install mongodb-community",
        ],
        start: ["brew services start mongodb/brew/mongodb-community"],
        docsUrl: "https://www.mongodb.com/docs/manual/tutorial/install-mongodb-on-os-x/",
      };
    case "rabbitmq":
      return {
        install: ["brew install rabbitmq"],
        start: ["brew services start rabbitmq"],
        postInstall: [
          "# Default guest user only works on localhost; the BrainAPI broker URL",
          "# expects kalo/kalo — create it with:",
          "rabbitmqctl add_user kalo kalo",
          "rabbitmqctl set_user_tags kalo administrator",
          "rabbitmqctl set_permissions -p / kalo \".*\" \".*\" \".*\"",
        ],
        docsUrl: "https://www.rabbitmq.com/install-homebrew.html",
      };
    case "milvus":
      return undefined;
    default:
      return undefined;
  }
}

function aptRecipe(service: ServiceName): NativeRecipe | undefined {
  switch (service) {
    case "redis":
      return {
        install: ["sudo apt-get update", "sudo apt-get install -y redis-server"],
        start: ["sudo systemctl enable --now redis-server"],
      };
    case "postgresql":
      return {
        install: [
          "sudo apt-get update",
          "sudo apt-get install -y postgresql-16 postgresql-16-pgvector",
        ],
        start: ["sudo systemctl enable --now postgresql"],
        postInstall: [
          "sudo -u postgres createuser --superuser \"$USER\"",
          "sudo -u postgres createdb \"$USER\"",
          "# Then update POSTGRES_USERNAME / POSTGRES_PASSWORD in .env",
        ],
        docsUrl: "https://www.postgresql.org/download/linux/ubuntu/",
      };
    case "neo4j":
      return {
        install: [
          "wget -O - https://debian.neo4j.com/neotechnology.gpg.key | sudo gpg --dearmor -o /etc/apt/keyrings/neotechnology.gpg",
          "echo 'deb [signed-by=/etc/apt/keyrings/neotechnology.gpg] https://debian.neo4j.com stable latest' | sudo tee /etc/apt/sources.list.d/neo4j.list",
          "sudo apt-get update && sudo apt-get install -y neo4j",
        ],
        start: ["sudo systemctl enable --now neo4j"],
        docsUrl: "https://neo4j.com/docs/operations-manual/current/installation/linux/debian/",
      };
    case "mongo":
      return {
        install: [
          "curl -fsSL https://www.mongodb.org/static/pgp/server-7.0.asc | sudo gpg -o /usr/share/keyrings/mongodb-server-7.0.gpg --dearmor",
          "echo 'deb [arch=amd64,arm64 signed-by=/usr/share/keyrings/mongodb-server-7.0.gpg] https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/7.0 multiverse' | sudo tee /etc/apt/sources.list.d/mongodb-org-7.0.list",
          "sudo apt-get update && sudo apt-get install -y mongodb-org",
        ],
        start: ["sudo systemctl enable --now mongod"],
        docsUrl: "https://www.mongodb.com/docs/manual/tutorial/install-mongodb-on-ubuntu/",
      };
    case "rabbitmq":
      return {
        install: ["sudo apt-get update", "sudo apt-get install -y rabbitmq-server"],
        start: ["sudo systemctl enable --now rabbitmq-server"],
        postInstall: [
          "sudo rabbitmqctl add_user kalo kalo",
          "sudo rabbitmqctl set_user_tags kalo administrator",
          "sudo rabbitmqctl set_permissions -p / kalo \".*\" \".*\" \".*\"",
        ],
      };
    case "milvus":
      return undefined;
    default:
      return undefined;
  }
}

function dnfRecipe(service: ServiceName): NativeRecipe | undefined {
  switch (service) {
    case "redis":
      return {
        install: ["sudo dnf install -y redis"],
        start: ["sudo systemctl enable --now redis"],
      };
    case "postgresql":
      return {
        install: [
          "sudo dnf install -y postgresql-server postgresql-contrib pgvector",
          "sudo /usr/bin/postgresql-setup --initdb",
        ],
        start: ["sudo systemctl enable --now postgresql"],
        docsUrl: "https://www.postgresql.org/download/linux/redhat/",
      };
    case "neo4j":
      return {
        install: [
          "sudo rpm --import https://debian.neo4j.com/neotechnology.gpg.key",
          "sudo dnf install -y neo4j",
        ],
        start: ["sudo systemctl enable --now neo4j"],
      };
    case "mongo":
      return {
        install: [
          "# Add MongoDB official repo first — see docs",
          "sudo dnf install -y mongodb-org",
        ],
        start: ["sudo systemctl enable --now mongod"],
        docsUrl: "https://www.mongodb.com/docs/manual/tutorial/install-mongodb-on-red-hat/",
      };
    case "rabbitmq":
      return {
        install: ["sudo dnf install -y rabbitmq-server"],
        start: ["sudo systemctl enable --now rabbitmq-server"],
        postInstall: [
          "sudo rabbitmqctl add_user kalo kalo",
          "sudo rabbitmqctl set_user_tags kalo administrator",
          "sudo rabbitmqctl set_permissions -p / kalo \".*\" \".*\" \".*\"",
        ],
      };
    case "milvus":
      return undefined;
    default:
      return undefined;
  }
}

function genericLinuxRecipe(
  service: ServiceName,
  pm: LinuxPackageManager,
): NativeRecipe | undefined {
  if (pm === "apt-get") return aptRecipe(service);
  if (pm === "dnf") return dnfRecipe(service);
  if (pm === "pacman") {
    switch (service) {
      case "redis":
        return {
          install: ["sudo pacman -S --noconfirm redis"],
          start: ["sudo systemctl enable --now redis"],
        };
      case "postgresql":
        return {
          install: [
            "sudo pacman -S --noconfirm postgresql",
            "# pgvector is available from the AUR: paru -S pgvector",
          ],
          start: [
            "sudo -iu postgres initdb -D /var/lib/postgres/data",
            "sudo systemctl enable --now postgresql",
          ],
        };
      default:
        return undefined;
    }
  }
  return undefined;
}

function windowsRecipe(service: ServiceName): NativeRecipe | undefined {
  switch (service) {
    case "redis":
      return {
        install: [
          "# Native Redis on Windows is no longer maintained.",
          "# Use Memurai (commercial), WSL, or Docker instead.",
        ],
        start: [],
        unsupported: "no first-party Windows build",
        docsUrl: "https://www.memurai.com/get-memurai",
      };
    case "postgresql":
      return {
        install: ["winget install PostgreSQL.PostgreSQL"],
        start: ["# postgresql-x64 service is auto-started by the installer"],
        docsUrl: "https://www.postgresql.org/download/windows/",
      };
    case "neo4j":
      return {
        install: ["winget install Neo4j.Neo4jDesktop"],
        start: [],
        docsUrl: "https://neo4j.com/download/",
      };
    case "mongo":
      return {
        install: ["winget install MongoDB.Server"],
        start: ["# MongoDB service is auto-started by the installer"],
      };
    case "rabbitmq":
      return {
        install: ["winget install RabbitMQ.RabbitMQ"],
        start: ["# RabbitMQ service is auto-started by the installer"],
        docsUrl: "https://www.rabbitmq.com/install-windows.html",
      };
    case "milvus":
      return undefined;
    default:
      return undefined;
  }
}

export function nativeRecipeFor(
  service: ServiceName,
  caps: PlatformCapabilities,
): NativeRecipe | undefined {
  if (service === "milvus") {
    return {
      install: [],
      start: [],
      unsupported:
        "Milvus has no native install path — use Docker or Zilliz Cloud (managed).",
      docsUrl: "https://milvus.io/docs/install_standalone-docker.md",
    };
  }
  if (caps.platform === "darwin" && caps.hasBrew) {
    const recipe = brewRecipe(service);
    if (recipe) return recipe;
  }
  if (caps.platform === "linux") {
    for (const pm of caps.linuxPackageManagers) {
      const recipe = genericLinuxRecipe(service, pm);
      if (recipe) return recipe;
    }
  }
  if (caps.platform === "win32") {
    const recipe = windowsRecipe(service);
    if (recipe) return recipe;
  }
  return undefined;
}
