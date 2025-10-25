variable "gcp_project"           { type = string }
variable "region"                { type = string }
variable "user_slug"             { type = string }   # e.g., "alice"
variable "plan"                  { type = string }   # "small" | "pro"
variable "version"               { type = string }   # e.g., "v0.4.2"
variable "domain"                { type = string }   # e.g., "example.com"
variable "zone_id"               { type = string }   # Cloudflare zone id
variable "ghcr_username"         { type = string }
variable "ghcr_token"            { type = string }   # GitHub PAT or fine-grained token
variable "base_image"            { type = string }   # e.g., "ghcr.io/lumen-labs/brainapi"
variable "enable_sql"            { type = bool   default = false }
variable "machine_type_small"    { type = string default = "e2-standard-2" }
variable "machine_type_pro"      { type = string default = "e2-standard-4" }
variable "subdomain_prefix"      { type = string default = "app" }  # app-<user>
variable "labels"                { type = map(string) default = {} }
