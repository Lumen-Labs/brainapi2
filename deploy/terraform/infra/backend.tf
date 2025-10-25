terraform {
  cloud {
    organization = "lumen-labs"
    workspaces { name = "brainapi-multi-tenant" }
  }
}
