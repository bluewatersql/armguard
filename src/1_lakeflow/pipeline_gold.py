"""Lakeflow SDP gold: appearance workload with own-baseline deltas, and pitcher-day model features."""
from pyspark import pipelines as dp

W = "PARTITION BY athlete_id ORDER BY game_date"
R3, B10 = (f"OVER ({W} ROWS BETWEEN {f})" for f in ("2 PRECEDING AND CURRENT ROW", "12 PRECEDING AND 3 PRECEDING"))


@dp.materialized_view(comment="Pitcher appearance; velo_delta = mean(last 3) - mean(prior 10) fastball mph; NULL if <5 prior")
def gold_appearances():
    return spark.sql(f"""
      WITH a AS (SELECT athlete_id, game_date, max(game_pk) game_pk, count(*) pitches, max(app_role) app_role,
                 coalesce(avg(CASE WHEN pitch_type IN ('FF','SI','FC') THEN release_speed END), avg(release_speed)) fb_velo,
                 avg(spin_rate) spin, avg(release_z) release_z, avg(extension) extension, min(temp_f) temp_f
                 FROM silver_pitches GROUP BY athlete_id, game_date),
      w AS (SELECT *, count(*) OVER ({W} ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) n_prior_apps,
            datediff(game_date, lag(game_date) OVER ({W})) rest_days,
            avg(fb_velo) {R3} velo_recent3, avg(fb_velo) {B10} velo_base10, avg(spin) {R3} - avg(spin) {B10} d_spin,
            avg(release_z) {R3} - avg(release_z) {B10} d_relz, avg(extension) {R3} - avg(extension) {B10} d_ext
            FROM a)
      SELECT *, IF(n_prior_apps >= 5, velo_recent3 - velo_base10, NULL) velo_delta,
             IF(n_prior_apps >= 5, d_spin, NULL) spin_delta, IF(n_prior_apps >= 5, d_relz, NULL) rel_z_delta,
             IF(n_prior_apps >= 5, d_ext, NULL) ext_delta FROM w""")


@dp.materialized_view(comment="Pitcher-day model features (spine 2025-03-27..09-28 x roster); label_il_14d = IL placement in next 14 days")
def gold_pitcher_daily_features():
    return spark.sql("""
      WITH spine AS (SELECT r.athlete_id, r.display_name, d.feature_date FROM silver_roster r
                     CROSS JOIN (SELECT explode(sequence(DATE'2025-03-27', DATE'2025-09-28')) feature_date) d),
      last_app AS (SELECT s.athlete_id, s.feature_date, a.velo_delta, a.spin_delta, a.rel_z_delta, a.ext_delta, a.n_prior_apps,
                   datediff(s.feature_date, a.game_date) days_since_app, CAST(a.app_role = 'SP' AS INT) role_sp
                   FROM spine s JOIN gold_appearances a ON a.athlete_id = s.athlete_id AND a.game_date <= s.feature_date
                   QUALIFY row_number() OVER (PARTITION BY s.athlete_id, s.feature_date ORDER BY a.game_date DESC) = 1),
      wl AS (SELECT s.athlete_id, s.feature_date,
             sum(CASE WHEN a.game_date > date_sub(s.feature_date, 7) THEN a.pitches ELSE 0 END) pitches_7d,
             sum(CASE WHEN a.game_date > date_sub(s.feature_date, 28) THEN a.pitches ELSE 0 END) pitches_28d,
             count(a.game_date) appearances_30d, min(CASE WHEN a.game_date > date_sub(s.feature_date, 7) THEN a.temp_f END) min_temp_7d
             FROM spine s LEFT JOIN gold_appearances a ON a.athlete_id = s.athlete_id
               AND a.game_date > date_sub(s.feature_date, 30) AND a.game_date <= s.feature_date GROUP BY 1, 2),
      stints AS (SELECT p.athlete_id, p.event_date placed, min(x.event_date) activated FROM silver_il_events p
                 LEFT JOIN silver_il_events x ON x.athlete_id = p.athlete_id AND x.action = 'activated' AND x.event_date > p.event_date
                 WHERE p.action = 'placed' GROUP BY 1, 2),
      il AS (SELECT s.athlete_id, s.feature_date,
             max(CASE WHEN s.feature_date >= t.placed AND s.feature_date < coalesce(t.activated, DATE'2099-01-01') THEN 1 END) IS NOT NULL on_il,
             max(CASE WHEN t.placed > s.feature_date AND t.placed <= date_add(s.feature_date, 14) THEN 1 END) IS NOT NULL label_il_14d
             FROM spine s LEFT JOIN stints t ON t.athlete_id = s.athlete_id GROUP BY 1, 2),
      sore AS (SELECT athlete_id, survey_date feature_date, avg(arm_soreness_0_10) OVER (PARTITION BY athlete_id ORDER BY survey_date
               ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) soreness_3d FROM silver_wellness)
      SELECT athlete_id, display_name, feature_date, velo_delta, spin_delta, rel_z_delta, ext_delta, n_prior_apps, days_since_app,
             role_sp, pitches_7d, pitches_28d, CAST(pitches_7d AS DOUBLE) / nullif(CAST(pitches_28d AS DOUBLE) / 4.0, 0) acwr,
             appearances_30d, min_temp_7d, soreness_3d, on_il, label_il_14d
      FROM spine LEFT JOIN last_app USING (athlete_id, feature_date) LEFT JOIN wl USING (athlete_id, feature_date)
      LEFT JOIN il USING (athlete_id, feature_date) LEFT JOIN sore USING (athlete_id, feature_date)""")


@dp.materialized_view(comment="Real league IL placements (2025), from MLB transactions")
def gold_league_il_stints():
    return spark.sql("SELECT player_id, team_id, txn_date placed_date FROM silver_transactions WHERE lower(description) LIKE '%placed%'")
