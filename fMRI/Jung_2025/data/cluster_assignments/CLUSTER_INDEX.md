# new54 cluster index — ID ↔ content ↔ figure

Reference for the 54-cluster (new54) partition that backs **all** Jung brain-validation
figures (Figs. 6 / D4 / D5). Every published map is a *single discovered cluster* from
this partition (App. D "14 clusters" is a typo for 54 — README §4).

## ⚠ Indexing: 0-based (IDs 0–53)

**Clusters are numbered 0…53.** The ID is identical across every layer — the model JSON
(`cluster_id`), the parsed CSV (`cluster_id`), the on-disk maps
(`group_cluster-NN_space-fsaverage6_…`, zero-padded), `config.FIGURES`, and the paper /
authors' cluster numbers. **There is no off-by-one: paper "cluster N" = code `cluster-0N`
= JSON `cluster_id N`, directly.**

Verified cross-check (2026-07-07): the authors state "cluster 7 is the only one not
significant." Our 0-based `group_cluster-07` has **0 FDR survivors** (maxT 3.68) while its
neighbours 5/6 are strong — so the authors' numbering lines up with the 0-based IDs exactly.

## Published clusters (the six figure networks)

| Cluster ID | CSV label (auto, video-name) | Network (authors' semantic label) | Figure | group-map maxT / FDR survivors (q<.05) |
|---|---|---|---|---|
| **5**  | `snakes` | **animals** (lateral PFC; snakes hunting reptile) | Fig. 6 / D4a | 4.73 / 78,169 |
| **32** | `mountainbike` | **natural landscapes** | Fig. 6 / D4b | 4.45 / 67,904 |
| **49** | `normativeprosocial1+2+3` | **faces** (prosocial human interaction → right IT near FFA) | Fig. D4c | 4.38 / 64,698 |
| **6**  | `planetearth` | animals (beach / rainforest) | Fig. D5 | 4.17 / 55,680 |
| **30** | `mountainbike` | natural landscapes | Fig. D5 | 4.32 / 65,655 |
| **31** | `mountainbike` | natural landscapes | Fig. D5 | 4.32 / 71,110 |

> The **faces** network is cluster **49**, *not* the `angrygrandpa` (#33–38) or
> `harrymetsally`-containing (#48) talking-head clusters — those are null (0 FDR
> survivors). Cluster 49's `normativeprosocial` videos show people interacting, which
> drives the face response. The CSV `cluster_label` is auto-derived from the dominant
> video filename (`derive_label`); the semantic "animals / landscapes / faces" names are
> the authors' (Slack 2026-07-07), not stored in the data.

## All 54 clusters (ID → CSV label → #clips)

| ID | label | #clips | | ID | label | #clips |
|---|---|---|---|---|---|---|
| 0 | stardust | 9 | | 27 | captureflag | 12 |
| 1 | stardust | 21 | | 28 | captureflag | 62 |
| 2 | stardust | 21 | | 29 | cyclegraphics | 70 |
| 3 | stardust | 47 | | **30** | **mountainbike** (D5) | 30 |
| 4 | wanderers | 98 | | **31** | **mountainbike** (D5) | 44 |
| **5** | **snakes** (D4a animals) | 61 | | **32** | **mountainbike** (D4b landscapes) | 31 |
| **6** | **planetearth** (D5) | 13 | | 33 | angrygrandpa | 13 |
| 7 | planetearth (null; not sig) | 61 | | 34 | angrygrandpa | 12 |
| 8 | dancewithdeath | 11 | | 35 | angrygrandpa | 8 |
| 9 | dancewithdeath | 10 | | 36 | angrygrandpa | 14 |
| 10 | dancewithdeath | 10 | | 37 | angrygrandpa | 54 |
| 11 | dancewithdeath | 37 | | 38 | angrygrandpa | 68 |
| 12 | alwaysafamily | 73 | | 39 | islamophobia | 10 |
| 13 | mixed_13 | 151 | | 40 | islamophobia | 111 |
| 14 | war | 15 | | 41 | giving+menrunning | 176 |
| 15 | war | 2 | | 42 | heartstop | 87 |
| 16 | war | 14 | | 43 | photography | 7 |
| 17 | forestfire | 12 | | 44 | photography | 13 |
| 18 | forestfire | 12 | | 45 | photography | 22 |
| 19 | carflood+tornado | 22 | | 46 | photography | 42 |
| 20 | huggingpets+lioncubs+universe | 108 | | 47 | gockskumara | 89 |
| 21 | HB | 5 | | **48** | beatbox+harrymetsally+unefille (null) | 206 |
| 22 | HB | 23 | | **49** | **normativeprosocial1+2+3** (D4c faces) | 131 |
| 23 | youth | 22 | | 50 | mediabias | 59 |
| 24 | youth | 24 | | 51 | gangan | 18 |
| 25 | mixed_25 | 235 | | 52 | gangan | 16 |
| 26 | captureflag | 15 | | 53 | gangan | 35 |

Total: 2,572 clip rows across 54 clusters. Regenerate this label/count table from the
vendored CSV with:

```bash
awk -F, 'NR>1{c[$1]++; l[$1]=$2} END{for(i=0;i<=53;i++) printf "%d\t%s\t%d\n", i, l[i], c[i]}' \
    cluster_assignments_new54clusters.csv
```
