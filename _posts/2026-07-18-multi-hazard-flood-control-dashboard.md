---
title: "Greater Manila Multi-Hazard & Flood Control Investment Dashboard"
date: 2026-07-18
categories: ['Project']
tags: ['Python', 'Streamlit', 'GeoPandas', 'Folium', 'Plotly', 'GIS', 'Spatial Analysis', 'Data Engineering', 'Public Policy']
math: true
---

<div style="width:100%; min-height:360px; display:flex; align-items:center; justify-content:center; border:2px dashed #ccc; border-radius:8px; background:#fafafa; margin-bottom:1.5rem;">
  <p style="color:#888; text-align:center; padding:1rem;">
    🎥 <em>Video placeholder: embed a walkthrough of the dashboard here (a YouTube iframe like the one in my previous post works well).</em>
  </p>
</div>

*Built together with Val Eltagonde.*

A hazard map is only as useful as the decision it can inform, and in the Philippines, the hazard maps and the infrastructure spending records live in two completely separate places. This means a planner who wants to know where risk is highest, and mitigation is weakest, has to cross-reference a NOAH shapefile against a DPWH project list manually, one municipality at a time. So Val Eltagonde and I built a dashboard that fuses them: it overlays PAGASA/Project NOAH's storm surge, flood, and landslide hazard layers with the Department of Public Works and Highways' flood control project records, scoped to the five provinces that make up the Greater Manila Area (Metro Manila, Bulacan, Cavite, Laguna, and Rizal, together known as "NCR Plus"). The [code is on GitHub](https://github.com/KGalapon/Multi-Hazard-Map-NOAH-Flood-Control-Projects); a public deployment is still in progress, so reach out at **karlchestergalapon77@gmail.com** if you would like early access.

As someone trying to think like a planner instead of just a data scientist, I kept returning to a few questions:

- Where in the Greater Manila Area is compound hazard risk of storm surge, flood, and landslide occurring together, highest?
- Which municipalities carry serious hazard exposure but little or no flood-control investment to show for it?
- Given the ₱122.9 billion already spent across 2,245 recorded projects in this region alone, is that spending actually going where the risk is?

Dashboards for hazard maps alone, or project spending alone, are not hard to find. However, I could not find one that put both in the same view, at the municipality level, with the compounding effect of overlapping hazards accounted for rather than averaged away. This turned out to be a GIS data engineering problem first and a modeling problem second, which isn't how the story usually goes on a project write-up like this one.

## Data Sources and Scope

Three sources feed the dashboard, and each one measures a different kind of ground truth. Project NOAH (Nationwide Operational Assessment of Hazards), maintained by the University of the Philippines, supplies the hazard geometry: separate shapefiles per province for storm surge, flood, and landslide, each polygon carrying a severity class. DPWH's flood control project records, released publicly on Kaggle, supply the money and location side: contract cost, dates, and coordinates for thousands of individual projects. GADM 4.1 supplies the administrative boundaries that tie the other two together. None of these three were built to be joined to each other, which is most of what the ETL step below actually deals with.

I also made a scoping decision early rather than defaulting to the whole country. A full nationwide rollout means 81 provinces and roughly 1,647 municipalities, and NOAH ships hazard files per province per hazard type, so that workload is not small. Instead, `config.FOCUS_NAME_1_SCOPE` narrows the ETL and the app to the Greater Manila Area's five provinces (113 municipalities) as the first working slice. This is not a shortcut baked into the code; it is a single config constant, so scaling back out to the whole country later means changing one line, not rewriting the pipeline.

## Building the Master Table

The ETL script (`etl/build_master_table.py`) is the only place spatial joins happen, and it exists specifically so the Streamlit app never has to touch raw geometry live. Everything the app reads comes from a precomputed Parquet table and a simplified GeoJSON, both regenerated only when the source data changes.

Most of the actual work in this step was reconciling three sources that disagreed with each other in small, specific ways. GADM only has two administrative levels for the Philippines (province and municipality), with no region layer at all, so PSA's 17-region grouping had to be attached through a hand-built crosswalk rather than read from any file. The DPWH CSV, meanwhile, has no status field: the original spec assumed Completed, Ongoing, and Proposed as options, but the real extract only has start and completion dates, and every row already has one, so every project is treated as completed. Projects are matched to municipalities by a point-in-polygon spatial join on each project's latitude and longitude, with a nearest-neighbor fallback for points that land just outside GADM's generalized coastline, rather than by matching the CSV's free-text municipality names against GADM's spelling. That one decision sidesteps an entire category of naming mismatches (Cotabato vs. North Cotabato, Metro Manila vs. Metropolitan Manila, and so on) that a text-matching join would have had to handle one exception at a time.

Hazard files stay unmerged by province rather than combined into one national file per hazard type, and this was a deliberate tradeoff rather than an oversight. Merging first would mean holding every province's polygon geometry in memory at once for no real benefit, since the overlay step only ever needs one province's file for that province's municipalities anyway. It also matches how NOAH actually distributes the data, so `find_province_hazard_file()` just matches a filename to a province by substring, case- and spacing-insensitive, and pulls in only what a given municipality needs. The cost is a bit more filename-matching logic in the ETL. What it buys is that a single corrected province file can be dropped in and re-run without touching, or risking, every other province's already-processed data.

## Scoring: The Composite Multi-Hazard Index

Each hazard layer gets overlaid with municipal boundaries, and every polygon in that layer carries a severity class of Low (1), Medium (2), or High (3). A municipality's score for one hazard is the area-weighted mean of that class across whatever portion of its area the hazard geometry actually covers, normalized to a 0-1 scale by dividing by the highest class present in that layer's own data (almost always 3, but detected per file instead of assumed). A municipality with zero hazard geometry for a given type is left undefined, not scored zero, and shown gray on the map: an inland town simply has not been assessed for storm surge, and scoring that a 0 would understate its true composite risk once the other hazards are averaged in. The one deliberate exception is storm surge in a landlocked province: Laguna and Rizal's only "coastline" is a lake, so a missing storm-surge file there is treated as a real, known 0 rather than a gap.

The three per-hazard scores then combine into one 0-100 Composite Multi-Hazard Index:

$$
\text{CMHI} = 100 \times \left(1 - \prod_{h \in H} (1 - s_h)\right)
$$

where $$H$$ is the set of hazards actually applicable to a municipality and $$s_h$$ is its normalized score for hazard $$h$$. This means CMHI behaves like a soft maximum rather than a plain average: a single dominant hazard pushes the score up close to its own value, and each additional concurrent hazard pushes it higher still instead of diluting it. A town scoring landslide at 0.9 but flood and storm surge at only 0.1 each would land around 37 under a plain average, understating that it has one genuinely dangerous hazard, but lands around 92 under this formula. Two simultaneous medium hazards (0.5 and 0.5) score 75, correctly worse than either alone. However, the formula has the mathematical shape of a probabilistic OR, which is a design choice, not a statistical claim: storm surge, flood, and landslide are not truly independent, since the same weather system often drives more than one at once in a coastal municipality. CMHI is meant to be read as a relative prioritization score, not a calibrated probability.

Hazard exposure alone does not tell you where to act, though, which is what the gap score is for:

$$
\text{gap\_score} = \text{CMHI} \times \frac{1}{1 + n_{\text{projects}}}
$$

A high-hazard municipality with zero existing projects surfaces at the top of this ranking, and even a single existing project meaningfully discounts the score rather than requiring several before the ranking responds. In the actual processed data, Doña Remedios Trinidad, Bulacan, currently holds the highest CMHI in the entire Greater Manila Area at roughly 37.6, with zero flood control projects recorded against it, so its gap score is simply its CMHI, unchanged. That is not a hypothetical edge case built to demonstrate the formula; it is the top row of the real ranking, and it is exactly the kind of municipality this dashboard exists to surface.

## What's in the Dashboard

The app itself is a single Streamlit page, but it is really several linked views over the same 113-municipality table:

- **Composite Multi-Hazard Risk choropleth**: the hero map, municipalities colored by CMHI, with gray meaning not-yet-assessed for at least one hazard rather than low risk. Clicking a municipality drives every panel below it.
- **Municipality Profile**: CMHI, per-hazard severity bands, project count, and total contract cost for whatever is currently selected.
- **Multi-Layer Hazard & Infrastructure Explorer**: a map clipped to the selected municipality or province, layering hazard polygons and project markers together, with a full-scope overlay available but off by default since the raw hazard layer alone runs into the tens of megabytes.
- **Priority Gap Ranking, Risk-vs-Coverage Quadrant, and Top Provinces/Municipalities by Hazard Index**: three tabs comparing hazard exposure against existing project coverage at the municipality, quadrant, and province level.
- **Regional roll-up**: a stacked severity breakdown per region, per hazard type.
- **Municipal Risk & Coverage Table**: the full filtered table with CSV export.
- **Methodology expander**: the formulas above plus a live worked example, so a reader never has to leave the dashboard to check how a number was produced.

![Composite Multi-Hazard Risk choropleth](/assets/Media/Multi_Hazard_Flood_Control_Dashboard/choropleth.png)
*Placeholder: screenshot of the CMHI choropleth map, Greater Manila colored by composite hazard score.*

![Priority Gap Ranking and Risk-vs-Coverage Quadrant](/assets/Media/Multi_Hazard_Flood_Control_Dashboard/priority-gap.png)
*Placeholder: screenshot of the Priority Gap Ranking tab, highlighting high-hazard, low-coverage municipalities like Doña Remedios Trinidad.*

## The Unglamorous Parts

The parts of this project that took the longest were not the scoring formula; they were keeping the app fast enough to actually use. `st.cache_data` is keyed off each source file's modification time, so re-running the ETL invalidates the app's cache automatically on next load, and this alone removed an entire category of "why is the dashboard showing stale numbers" debugging. However, caching does nothing about geometry that is simply too dense to draw in a browser tab, and NOAH's flood and landslide polygons are considerably denser than GADM's admin boundaries. `config.SIMPLIFY_TOLERANCE` controls display resolution for both the municipality choropleth and the hazard explorer layer, and getting that number right was mostly trial and error: my own processed-data folder still has `hazard_display_layers_simplified_final.geojson`, `hazard_display_layers_simplified_thrice.geojson` (which went the wrong direction, at 65MB), and the file the app actually reads today, `hazard_display_layers_simplified_FINALFINAL.geojson`, at a much more browser-friendly 5.5MB. That naming is embarrassing and also completely honest about what iterating on a simplification tolerance actually looks like in practice. Crucially, none of this touches the geometry used to compute `hazard_score_<type>` itself; simplification is a display-only concern, kept deliberately separate from anything the CMHI formula depends on.

## Points for Improvement and Future Direction

The most obvious next step is also the one already designed for: extending `FOCUS_NAME_1_SCOPE` back to the full 81 provinces and roughly 1,647 municipalities nationwide. The ETL and the app do not need to change to do this, but the hazard files do need to actually be available at that scale, and the geometry-simplification struggle above suggests the memory and rendering budget deserves real testing before flipping that switch. A second, related step is getting the app onto a public URL instead of running it locally; that has mostly been blocked on finishing the hazard-data ingestion for every province rather than any deployment obstacle. Beyond these, the equal-weighting assumption behind CMHI (each hazard counts 1/3 regardless of local context) is a reasonable default but not one I have stress-tested; a province where landslide risk genuinely dominates flood risk might deserve a different weighting than the current formula gives it, and the "Advanced" weighting recompute already exposed in the app is the natural place to explore that.

## Learnings

Working through this taught me more about spatial data engineering than about machine learning, which was not what I expected going in:

- performing spatial joins and area-weighted overlays with **GeoPandas** and **Shapely**
- reconciling three real-world datasets with schemas that quietly disagreed with a written spec
- separating an offline ETL step from a caching, read-only app, instead of computing anything live
- choosing a geometry-simplification tolerance for web maps without silently corrupting the numbers that formulas depend on
- designing a composite scoring formula (complement-product aggregation) that behaves sensibly under missing data instead of just failing on it

## Closing Thoughts

None of this is useful if it stays a personal side project. If this dashboard, or the reasoning behind it, ever reaches someone who actually decides where the next flood-control peso goes, that is the only measure of success I care about here.

This dashboard was built by Karl Galapon and Val Eltagonde.

Thank you for reading!
