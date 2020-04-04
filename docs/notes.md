Opisy eksperymentów:

- rnn_v1 - 3xlstm, 3xDense
- rnn_v2 - 2xlstm, 2xDense
- rnn_v0 - 1xlstm, 1xDense
- chinit_e1_p1 - inicjalizacja na checkpointach

* chinit_e4_p1, chinit_e5_p1 - od wszystkich poprzednich wyników
* chinit_e6_p1, chinit_e7_p1 - od 100 poprzednich wyników

* chin_e1_p1 - zmniejszanie checkpointów zaczyna się od 0.5 średniej nagrody ze 100 ostatnich gier, zmiejszane jest po 0.001 jednostkę na grę ponad tym progiem
* chin_e2_p1 - jak wyżej, ale po 0.0001 jednostkę

* chin_e7_p1 - 1000
* chin_e8_p1 - 300
* chin_e9_p1 - 300, inicjalizacja v3.0

* scon_e1_p1 - 0 to 1 to 2 to 5 - dotrenowanie
* scon_e3_p1 - 0 to 1 to 2 to 5 - dotrenowanie
* scon_e3_p2* - from scon_e3_p1
    * hard - difficuly from 0.6 to 1.0
    * nhm - do not handle the iterations manually
* scon_e3_p3_hard - from scon_e3_p2 (after it was stopped)

* m5v0_e1_p1 - initialization 1 net 4 times against 0 enemies
* m5v1_e1_p2* - transfer to 1 enemy
    * ovf - after ~1.4B steps, overfitted (running right)
    * part - after ~400M steps, not overfitted, but mostly scoring
