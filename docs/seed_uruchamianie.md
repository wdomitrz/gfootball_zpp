# Tutorial seed
1. instalujemy docker
2. dodajemy się do grupy docker `usermod -aG docker $USER`
   (ważne jak ktoś odpala z sudo (czyli jako inny user) to dostanie od gcloud)
	`
	unauthorized: You don't have the needed permissions to perform this operation,
	and you may have invalid credentials. To authenticate your request, follow the
	steps in:
	https://cloud.google.com/container-registry/docs/advanced-authentication
	`
	(zatem nie odpalamy z sudo)
3. relogin/reboot
4. jeżeli nie mamy `gcloud` to instalujemy
	* https://cloud.google.com/sdk/install
4. `gcloud auth login`
5. `gcloud config set project warsaw-zpp`
6. `gcloud auth configure-docker`
	* wybieramy Y
8. zmieniamy buckety
	* https://github.com/google-research/seed_rl/blob/08fef21e1aa46f399ce225d3fdbb8f8cd737e7b8/gcp/setup.sh#L25
	* https://github.com/google-research/seed_rl/blob/08fef21e1aa46f399ce225d3fdbb8f8cd737e7b8/gcp/setup.sh#L30
9. odpalamy
10. jeżeli nie działa to `gcloud auth print-access-token | docker login -u oauth2accesstoken --password-stdin https://gcr.io`

# seed na gcp
* `./cluster_make.sh skonrad 1 start; ./cluster_make.sh skonrad 1 fix_firewall; ./cluster_make.sh skonrad 1 ssh;`
* `./moj_seed.sh nick` (zmieni nazwy bucketow na `nick_seed_rl`; nazwe job też)
* `cd nick_seed_rl`

# tensorboard
* `gcloud auth application-default login`
* `tensorboard --logdir=gs://skonrad_seed_rl`

# Zmienianie regionu uruchamiania seed-a
* `gcloud compute regions list` listuje dostępne regiony z zajęciem i dostępnością zasobów,
* skrypt `run.sh` ma 5., opcjonalny parametr - nazwę regionu z domyślną wartością `us-central1` (tak, jak teraz zawsze). Jeśli zostanie podana odpowiednia nazwa regionu, to seed zostanie uruchomiony w tym regionie.
