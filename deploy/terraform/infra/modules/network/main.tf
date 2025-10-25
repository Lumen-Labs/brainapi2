resource "google_compute_network" "vpc" {
  name                    = "${var.name_prefix}-vpc"
  auto_create_subnetworks = false
}

resource "google_compute_subnetwork" "subnet" {
  name          = "${var.name_prefix}-subnet"
  ip_cidr_range = "10.42.0.0/24"
  region        = var.region
  network       = google_compute_network.vpc.id
}

resource "google_compute_firewall" "allow_http" {
  name    = "${var.name_prefix}-allow-http"
  network = google_compute_network.vpc.name

  allow { protocol = "tcp", ports = ["80", "443", "8080", "22"] }
  source_ranges = ["0.0.0.0/0"]
}

output "subnet_selflink" { value = google_compute_subnetwork.subnet.self_link }
