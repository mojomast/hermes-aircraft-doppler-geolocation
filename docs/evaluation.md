# Evaluation

How the real-world numbers in the README were produced. Harness: `scripts/eval_batch.py` (manifest format and flags are in the README, "Evaluation harness"). The audio and the manifest are not in the repo: the audio stays with its owners. Clip IDs, sources and licences are listed below.

## Datasets

| set | what | clips (used / excluded) | splits |
| --- | --- | --- | --- |
| Original (chunks 1–4) | 12 airport-spotting YouTube clips and livestream segments (CRQ, DAB, VNY, LGB, ADS, OSH) | 12 / 0 | 7 tune, 5 test |
| Livestreams (chunk 5) | 25 livestream VODs filmed at known spotting locations | 21 / 4 | 16 tune, 9 test2 |
| Field recordings (chunks 6–7) | raw, dated, geotagged Freesound and radio aporee ::: maps recordings with no commentary | 14 / 7 | tune6, test6 |

Exclusions are recorded with reasons in the manifest. They were: diagnosed narration-dominated, high-passed or ATC-gated livestream audio (4, tune only, after diagnosis); no recording time; no adsb.lol data for the day (2023-09-14 and 2024-04-17 releases have no tar parts, 2025-08-17 had no coverage near Whitehorse, 2025-11-10 timed out twice); and no aircraft within 12 km during the clip.

## Split policy

- Clips are split **by location** before any tuning, so the held-out set has no airport or spot in common with tune. The test2 split (ATL DFW DUB DUS LGW PHX SXM SYD) and the test6 split (MUC EDI HAG, plus clips later excluded) were fixed before any run on them.
- Settings (`sigma_model`, `detect_prob`, `temper_extra`, the broadband gates) were tuned on tune only. A held-out split is run **once** with frozen code and settings, and the result is reported whatever it is.
- One change was made before any broadband run: `mco_0605` moved from test6 to tune6 for chunk 7.
- Every run searches a 15 km radius around a centre offset 5 km from the truth, at a bearing derived from the clip id, so the grid is never aligned on the answer. Tolerance is ±30 s on the start time, plus ±600 s for some original clips, and dab_part1 uses a fixed ±9000 s.

## Ground truth

- **Location.** Livestreams and spotting videos: the spot named in the title or description, or visible on camera (runway end, park, viewing area), with `truth_unc_m` set from how precisely it is known (300–1500 m). Field recordings: the recordist's own map pin on Freesound or radio aporee. `loc_evidence` records the reasoning per clip.
- **Time.** Livestreams: YouTube `release_timestamp` (stream start) plus the segment offset, cross-checked against on-screen clocks where shown. Field recordings: the recordist-entered date and time (aporee `recdate` with its IANA zone; Freesound title stamps or descriptions), converted to UTC. `time_evidence` records the source.
- **ADS-B sanity check (field recordings).** Before any run, ADS-B passes near the pin (< 4 km and < 12 km) were lined up against loudness peaks in the audio. For example, man_hg0903 passes at 0.5 / 3.3 / 8.1 / 15.3 min match audio peaks at 0.4 / 3.3 / 8.1 / 15.2 min. man_hg0909's date was resolved this way from a contradictory title. Clips with no aircraft in the clip were excluded, not counted as failures.
- **Error.** The great-circle distance from the best grid point to the truth. "In 95 %" (`in95u`) means the truth lies inside the 95 % credible area padded by `truth_unc_m`.

## Results

### Tonal pipeline (0.2.0 defaults, ±30 s)

| | runs | no events / no ADS-B | solved | error < 3 km | median error (solved) | truth in padded 95 % |
|---|---|---|---|---|---|---|
| Tune | 24 | 11 / 1 | 12 | 5 | 7.2 km | 9 / 12 |
| Held-out (14 clips) | 14 | 6 / 0 | 8 | 1 | 12.1 km | 6 / 8 |
| Field tune6 (±30 s and ±120 s) | 14 | 12 / 0 | 2 | 0 | 10.5 and 19.8 km | — |
| Field test6, once (±30 s and ±120 s) | 8 | 8 / 0 | 0 | 0 | — | — |

Confidence flag: on tune, the `ok` runs were right (< 3 km) 3 of 3 times. On held-out, the only `ok` run (osh_0719) was 13.2 km off.

### Broadband events (0.3.0, `--broadband` / `--det broadband=true`, frozen at ff25280, ±30 s)

| clip | before (tonal only) | after |
|---|---|---|
| man_hg0903 | 10.5 km, low | **0.19 km, ok** (27 events, 26 consistent) |
| man_hg0909 | no events | **0.33 km, ok** (4 events) |
| ams_pb_M1Qg | no events | 4.6 km, low (1 event) |
| mco_0605, bru_ap77758, bour_fs839494 | no events | no events |
| vny_0513 | 15.4 km, low | 2.95 km, ok |
| ord_2H7D | 15.2 km, low | 5.9 km, ok |
| jfk_rgow | 10.8 km, low | 4.6 km, ok |
| crq | 1.16 km, ok | 1.55 km, ok |
| lgb_0810, vny_0605, ads_0821a, dab_0820a, dab_0820b, mia_7Nyj, bur_l4OP | — | unchanged |

**Held-out (one frozen run at ff25280, `--det broadband=true`, ±30 s; dab_part1 ±9000 s).** 18 clips: 5 original test, 9 livestream test2, 4 field-recording test6. "Before" is the tonal-only 0.2.0 run (c5_heldout / c6_test); muc_ap82011 had no tonal-only run because its ADS-B day arrived later.

| clip | split | before | after (broadband on) | truth in padded 95 % |
|---|---|---|---|---|
| dfw_iNsj | test2 | no events | **1.96 km**, low | yes |
| lgb_0807 | test | 2.44 km, low | **2.38 km, ok** | yes |
| dub_or3L | test2 | no events | 4.74 km, ok | yes |
| lgw_1pyp | test2 | 5.14 km, low | 5.14 km, low | yes |
| muc_ap82011 | test6 | — | 13.07 km, ok | **no** |
| osh_0719 | test | 13.24 km, ok | 13.24 km, ok | **no** |
| ads_0821b | test | 7.88 km, low | 17.24 km, low (regression) | yes |
| dab_part1 | test | 17.87 km | 17.96 km, low | yes |
| syd_DuZb | test2 | 11.01 km, low | 18.58 km, low (regression) | **no** |
| atl_0IA3 | test2 | 18.61 km, low | 18.61 km, low | yes |
| vny_0526 | test | 19.69 km, low | 19.69 km, low | yes |
| sxm_4e38, dus_3afV, phx_L2dM, phx_rnu4 | test2 | no events | no events | — |
| muc_ap81998, edi_ap82914, hag_ap83009 | test6 | no events | no events | — |

| | runs | solved | error < 3 km | median error (solved) | truth in padded 95 % |
|---|---|---|---|---|---|
| Tonal only (0.2.0), 14 clips | 14 | 8 | 1 | 12.1 km | 6 / 8 |
| **Broadband on (0.3.0), 18 clips** | 18 | 11 | **2** | **13.2 km** | 8 / 11 |

Honest reading: broadband events add solved runs (dfw and dub went from no events to an answer) and one more run under 3 km, but the median of solved runs got slightly worse, and two clips regressed (ads_0821b 7.9 → 17.2 km, syd 11.0 → 18.6 km). Of the four `ok` results, lgb (2.4 km) and dub (4.7 km) are roughly right, while osh (13.2 km) and muc_ap82011 (13.1 km) are wrong, and their 95 % areas miss the truth. The field recordings at Edinburgh, The Hague and Munich north, and most livestreams, still give no events. On unseen clips the method still mostly fails.

Noise-only control audio (600 s) gives 0 broadband events.

## Recordings and attribution

Field recordings (Freesound CC0, radio aporee ::: maps public domain; "in manifest" means usable, not that every clip has been run yet; the CC BY clips were excluded before evaluation and none of their audio or derived images is published):

| id | split | date | recordist / source | licence | status |
|---|---|---|---|---|---|
| `man_hg0903` | tune6 | 2026-09-03 | [Sadiquecat (freesound.org/s/868926)](https://freesound.org/people/Sadiquecat/sounds/868926/) | CC0 | in manifest |
| `man_hg0909` | tune6 | 2026-09-09 | [Sadiquecat (freesound.org/s/869798)](https://freesound.org/people/Sadiquecat/sounds/869798/) | CC0 | in manifest |
| `gergueil_0427` | tune6 | 2025-04-27 | [Sadiquecat (freesound.org/s/802703)](https://freesound.org/people/Sadiquecat/sounds/802703/) | CC0 | in manifest |
| `yxy_0417` | test6 | 2024-04-17 | [SoundsLikeYukon (freesound.org/s/732866)](https://freesound.org/people/SoundsLikeYukon/sounds/732866/) | CC0 | excluded: adsb.lol 2024-04-17 release has no tar parts |
| `yxy_0817` | test6 | 2025-08-17 | [SoundsLikeYukon (freesound.org/s/820320)](https://freesound.org/people/SoundsLikeYukon/sounds/820320/) | CC0 | excluded: ADS-B day 2025-08-17 has 0 aircraft in the Whitehorse box (no receiver coverage) |
| `mco_0605` | tune6 | 2024-06-05 | [joeffl (freesound.org/s/843280)](https://freesound.org/people/joeffl/sounds/843280/) | CC0 | in manifest |
| `nl_klank_0914` | tune6 | 2023-09-14 | [klankbeeld (freesound.org/s/703104)](https://freesound.org/people/klankbeeld/sounds/703104/) | CC-BY-4.0 | excluded: adsb.lol 2023-09-14 release has no tar parts |
| `som_0829` | test6 | 2025-08-29 | [curious_world_on_youtube (freesound.org/s/823581)](https://freesound.org/people/curious_world_on_youtube/sounds/823581/) | CC-BY-4.0 | excluded: ADS-B sanity: no aircraft within 12 km during clip (night, 9 tracks) |
| `muc_ap82011` | test6 | 2025-11-19 | [Jan Krtička (radio aporee ::: maps, loc 70347)](https://aporee.org/maps/?loc=70347) | PUBLIC-DOMAIN | in manifest |
| `muc_ap81998` | test6 | 2025-11-22 | [Jan Krtička (radio aporee ::: maps, loc 70334)](https://aporee.org/maps/?loc=70334) | PUBLIC-DOMAIN | in manifest |
| `muc_ap82012` | test6 | 2025-11-10 | [Jan Krtička (radio aporee ::: maps, loc 70348)](https://aporee.org/maps/?loc=70348) | PUBLIC-DOMAIN | excluded: ADS-B day 2025-11-10 fetch timed out twice |
| `edi_ap82914` | test6 | 2025-09-17 | [Boris Allenou (radio aporee ::: maps, loc 71095)](https://aporee.org/maps/?loc=71095) | PUBLIC-DOMAIN | in manifest |
| `hag_ap83009` | test6 | 2026-03-03 | [nicozentz (radio aporee ::: maps, loc 71173)](https://aporee.org/maps/?loc=71173) | PUBLIC-DOMAIN | in manifest |
| `bru_ap77758` | tune6 | 2025-04-21 | [Flavien Gillié (radio aporee ::: maps, loc 67216)](https://aporee.org/maps/?loc=67216) | PUBLIC-DOMAIN | in manifest |
| `bru_ap82700` | tune6 | 2026-02-08 | [Flavien Gillié (radio aporee ::: maps, loc 70927)](https://aporee.org/maps/?loc=70927) | PUBLIC-DOMAIN | in manifest |
| `lon_vp_ap85488` | tune6 | 2026-08-19 | [David Pitt (radio aporee ::: maps, loc 73201)](https://aporee.org/maps/?loc=73201) | PUBLIC-DOMAIN | in manifest |
| `pln_ap79888` | tune6 | 2025-07-18 | [Claire Sauvaget (radio aporee ::: maps, loc 68865)](https://aporee.org/maps/?loc=68865) | PUBLIC-DOMAIN | excluded: ADS-B sanity: no aircraft within 12 km during clip |
| `fai_ap84541` | tune6 | 2026-06-07 | [sjdowgray (radio aporee ::: maps, loc 72394)](https://aporee.org/maps/?loc=72394) | PUBLIC-DOMAIN | in manifest |
| `ram_fs807178` | tune6 | 2025-04-27 | [sounds_from_palestine (freesound.org/s/807178)](https://freesound.org/people/sounds_from_palestine/sounds/807178/) | CC0 | in manifest |
| `bour_fs839494` | tune6 | 2025-12-21 | [nicotep (freesound.org/s/839494)](https://freesound.org/people/nicotep/sounds/839494/) | CC0 | in manifest |

YouTube (all rights reserved by their channels; audio was analysed locally and is not redistributed): `crq` ([video](https://www.youtube.com/watch?v=-BbVKXQ5rtM)), `dab_part1` ([video](https://www.youtube.com/watch?v=k-ZEf7kPYmQ)), `vny_0605` ([video](https://www.youtube.com/watch?v=8yjuZuxj6Ss)), `vny_0513` ([video](https://www.youtube.com/watch?v=JymUmuzBpNI)), `vny_0526` ([video](https://www.youtube.com/watch?v=azRssSWa9mk)), `lgb_0810` ([video](https://www.youtube.com/watch?v=3LrcOZ3Whpo)), `lgb_0807` ([video](https://www.youtube.com/watch?v=5pKbyaaROIE)), `ads_0821a` ([video](https://www.youtube.com/watch?v=N_4WSZIYiz4)), `ads_0821b` ([video](https://www.youtube.com/watch?v=7cjlMGoSuuQ)), `dab_0820a` ([video](https://www.youtube.com/watch?v=_yUtFSsZI9A)), `dab_0820b` ([video](https://www.youtube.com/watch?v=v9ws1TVTcm0)), `osh_0719` ([video](https://www.youtube.com/watch?v=WOFnYOk5HL0)), `sxm_4e38` ([video](https://www.youtube.com/watch?v=4e386QfQFrw)), `syd_DuZb` ([video](https://www.youtube.com/watch?v=DuZbEXyqjiU)), `man_CSCQ` ([video](https://www.youtube.com/watch?v=CSCQBm61UDg)) (excluded), `ams_pb_M1Qg` ([video](https://www.youtube.com/watch?v=M1Qg3klWNmY)) (excluded), `fra_zh_-THx` ([video](https://www.youtube.com/watch?v=-THxVD2O_AM)), `dfw_iNsj` ([video](https://www.youtube.com/watch?v=iNsjLnf0tUI)), `bos_IzU2` ([video](https://www.youtube.com/watch?v=IzU2qKbzKvI)), `yyz_tw_k2cn` ([video](https://www.youtube.com/watch?v=k2cnaaE0lnM)), `yyz_fd_CXXk` ([video](https://www.youtube.com/watch?v=CXXkDITYrWk)), `san_2btr` ([video](https://www.youtube.com/watch?v=2btr0C8Onwo)) (excluded), `san_YBY5` ([video](https://www.youtube.com/watch?v=YBY5LQ12xMA)), `mia_7Nyj` ([video](https://www.youtube.com/watch?v=7Nyjg7AmxEQ)), `ord_6Yik` ([video](https://www.youtube.com/watch?v=6YikpZR-gHo)), `ord_2H7D` ([video](https://www.youtube.com/watch?v=2H7DOvZqhcw)), `dus_3afV` ([video](https://www.youtube.com/watch?v=3afVaUF1nAc)), `sea_BDNb` ([video](https://www.youtube.com/watch?v=BDNbYDuUHzY)) (excluded), `phx_L2dM` ([video](https://www.youtube.com/watch?v=L2dMARcIy7g)), `phx_rnu4` ([video](https://www.youtube.com/watch?v=rnu4UQFUb9c)), `bur_l4OP` ([video](https://www.youtube.com/watch?v=l4OPhA7LfAA)), `atl_0IA3` ([video](https://www.youtube.com/watch?v=0IA3JF6xT9w)), `jfk_rgow` ([video](https://www.youtube.com/watch?v=rgowZUyfzPg)), `lgw_1pyp` ([video](https://www.youtube.com/watch?v=1pypl-EYPLw)), `dub_or3L` ([video](https://www.youtube.com/watch?v=or3LZ4rYnFM)), `mel_JWvE` ([video](https://www.youtube.com/watch?v=JWvEcbzMi9s)), `mel_rOCY` ([video](https://www.youtube.com/watch?v=rOCYJD89Eh0)), `cfu_QUWR` ([video](https://www.youtube.com/watch?v=QUWReiFynRA)) (excluded).
