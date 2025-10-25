resource "google_compute_instance" "vm" {
  name         = "${var.name_prefix}-vm"
  machine_type = var.machine_type
  zone         = "${var.region}-a"

  boot_disk {
    initialize_params {
      image = "ubuntu-os-cloud/ubuntu-2204-jammy-v20241001"
      size  = 30
      type  = "pd-balanced"
    }
  }

  network_interface {
    subnetwork = var.subnet_selflink
    access_config {} # ephemeral public IP
  }

  metadata = {
    user-data = templatefile("${path.module}/templates/cloud-init.yaml.tftpl", {
      fqdn        = var.fqdn
      image_ref   = var.image_ref
      ghcr_user   = var.ghcr_user
      ghcr_token  = var.ghcr_token
      enable_sql  = var.enable_sql
      db_conn_str = var.db_conn_str
    })
  }

  labels = var.labels
}

output "public_ip" {
  value = google_compute_instance.vm.network_interface[0].access_config[0].nat_ip
}
