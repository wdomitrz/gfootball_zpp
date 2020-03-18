## GCP
1. Robimy swoją maszynę ze snapshoota `skonrad-local-seed-21`
2. Można użyć skryptów do przesłania danych lub zrobić samemu to co one.
   Przykładowo:
	+ `/gcp_connect.sh user_setup skonrad-local-seed-2 skonrad us-west1-b`
		- montuje home zdalnej maszyny korzystając z sshfs
		- klonuje tam nasze repo seed_rl i gfootball_zpp
		- odmontowuje katalog zdalny
		- wolny bo odpala gita lokalnie
		- w gfootball_zpp powinno się zrobić checkout na `run_local`
		  (o ile nie zostało już włączone do mastera)
	+ `/gcp_connect.sh enter skonrad-local-seed-2 skonrad us-west1-b`
		- montuje home zdalnej maszyny korzystając z sshfs
		- wykonuje ssh na zdalną maszynę
3. Plik `install_dependencies.txt` może się przydać jeżeli ktoś chce postawić u siebie
   na komputerze (wymaga pobrania nvidia nccl (patrz komentarze w pliku))
## Odpalanie lokalne
* `run_local.sh AIP_CONFIG_FILE ENV_CONFIG_FILE NUM_ACTORS`
* Uwagi
	+ należy pamiętać aby `NUM_ACTORS` pasowało do `inference_batch_size` z
      `AIP_CONFIG_FILE` (było odpowiednio duże 
	  lub `inference_batch_size` odpowiednio małe)
	+ karty w tmux przełączamy `ctrl+b numer_karty`
	+ scroll w tmux włączamy `ctrl+b [`
