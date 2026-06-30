# 2026 3rd Infodengue-Mosqlimate Dengue Challenge (IMDC) — Epidemáticos

Submission repositories for the **2026 3rd Infodengue-Mosqlimate Dengue Challenge**: weekly probable-case forecasts for **dengue** and **chikungunya** in Brazil, at both state and city level.

## Team and Contributors

**Team:** Epidemáticos — School of Applied Mathematics, Fundação Getulio Vargas (FGV/EMAp), Rio de Janeiro, Brazil

- [Eduardo Adame, M.Sc. — FGV/EMAp](https://github.com/adamesalles)
- [Ezequiel Braga, M.Sc. — FGV/EMAp](https://github.com/ezequielebs)
- [Iara Cristina, M.Sc. — FGV/EMAp](https://github.com/iaracastro)
- [Isaque Pim, Ph.D. — FGV/EMAp](https://github.com/isaquepim)

## Repositories

The team's submission is split across three model repositories, all sharing the same raw and processed data (in `raw_data/` and `processed_data/` here) and data-preparation pipeline (in `data_prep/`):

| Repository | Model | Scope |
|---|---|---|
| [3rd_imdc_emap_epidematicos_prophet](https://github.com/EzequielEBS/3rd_imdc_emap_epidematicos_prophet) | Prophet | State & city level (dengue + chikungunya) |
| [3rd_imdc_emap_epidematicos_sarimax_state](https://github.com/EzequielEBS/3rd_imdc_emap_epidematicos_sarimax_state) | SARIMAX | State level (dengue + chikungunya) |
| [3rd_imdc_emap_epidematicos_sarimax_muni](https://github.com/EzequielEBS/3rd_imdc_emap_epidematicos_sarimax_muni) | SARIMAX | Municipality level / Optional City-Level Challenges (dengue + chikungunya) |

## Repository Structure

```
.
├── README.md
├── LICENSE
├── .gitattributes
│
├── data_prep/              # shared R data-preparation pipeline
├── raw_data/               # original input datasets (cases, climate, population, …)
└── processed_data/         # modeling-ready tables produced by data_prep/
```

Each model repository contains its own `README.md` with full details on team, data, model training, data-usage restrictions, predictive uncertainty, and references, as required by the IMDC submission format.
