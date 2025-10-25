variable "name_prefix"     { type = string }
variable "region"          { type = string }
variable "subnet_selflink" { type = string }
variable "machine_type"    { type = string }
variable "fqdn"            { type = string }
variable "image_ref"       { type = string }
variable "ghcr_user"       { type = string }
variable "ghcr_token"      { type = string }
variable "labels"          { type = map(string) default = {} }
variable "enable_sql"      { type = bool }
variable "db_conn_str"     { type = string, default = null }
