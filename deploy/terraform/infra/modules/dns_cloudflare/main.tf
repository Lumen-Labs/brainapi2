variable "zone_id"  { type = string }
variable "name"     { type = string }
variable "domain"   { type = string }
variable "target_ip"{ type = string }
variable "ttl"      { type = number default = 120 }

resource "cloudflare_record" "a" {
  zone_id = var.zone_id
  name    = "${var.name}.${var.domain}"
  type    = "A"
  value   = var.target_ip
  ttl     = var.ttl
  proxied = true
}

output "fqdn" { value = cloudflare_record.a.hostname }
