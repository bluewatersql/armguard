"""Lakeflow SDP bronze+silver: Auto Loader over the UC Volume, typed/deduped silver with expectations."""
from pyspark import pipelines as dp
from pyspark.sql import functions as F

L = spark.conf.get("landing")


def auto(path, variant):
    r = spark.readStream.format("cloudFiles").option("cloudFiles.format", "json")
    r = r.option("singleVariantColumn", "payload") if variant else r.option("cloudFiles.inferColumnTypes", "true")
    return r.load(f"{L}/{path}/").withColumn("src_file", F.col("_metadata.file_path")).withColumn("ingested_at", F.current_timestamp())


@dp.table(comment="Raw MLB GUMBO game feeds (VARIANT); statsapi.mlb.com")
@dp.expect_or_fail("payload_present", "payload IS NOT NULL")
def bronze_mlb_games():
    return auto("mlb_games", True)


@dp.table(comment="Raw MLB transactions (VARIANT); statsapi.mlb.com")
@dp.expect_or_fail("payload_present", "payload IS NOT NULL")
def bronze_mlb_transactions():
    return auto("mlb_transactions", True)


def club(n):  # closure factory: each bronze_club_* binds its own feed name
    @dp.table(name=f"bronze_club_{n}", comment=f"Raw club {n} feed (JSON lines)")
    def t():
        return auto(f"club_{n}", False)


for n in ("tracking", "wellness", "trainer_notes", "medical", "roster"):
    club(n)


@dp.materialized_view(comment="One row per MLB game (latest landed feed wins)")
@dp.expect_or_drop("game_pk_present", "game_pk IS NOT NULL")
def silver_games():
    return spark.sql("""
      SELECT payload:gamePk::bigint game_pk, to_date(payload:gameData.datetime.officialDate::string) game_date,
             payload:gameData.teams.home.id::int home_id, payload:gameData.teams.away.id::int away_id,
             payload:gameData.venue.name::string venue, try_cast(payload:gameData.weather.temp::string AS INT) temp_f,
             payload:gameData.gameInfo.attendance::int attendance
      FROM bronze_mlb_games
      QUALIFY row_number() OVER (PARTITION BY payload:gamePk::bigint ORDER BY ingested_at DESC) = 1""")


@dp.table(comment="Club pitch tracking, typed and deduped on pitch_uid")
@dp.expect_or_drop("velo_plausible", "release_speed BETWEEN 40 AND 110")
@dp.expect("spin_plausible", "spin_rate IS NULL OR spin_rate BETWEEN 0 AND 4000")
def silver_pitches():
    return spark.readStream.table("bronze_club_tracking").selectExpr(
        "pitch_uid", "CAST(game_pk AS BIGINT) game_pk", "to_date(game_date) game_date", "athlete_id", "app_role",
        "pitch_type", "CAST(release_speed AS DOUBLE) release_speed", "CAST(spin_rate AS DOUBLE) spin_rate",
        "CAST(release_z AS DOUBLE) release_z", "CAST(extension AS DOUBLE) extension", "CAST(temp_f AS INT) temp_f",
    ).dropDuplicates(["pitch_uid"])


@dp.materialized_view(comment="League injured-list status changes (typeCode SC)")
def silver_transactions():
    return spark.sql("""
      SELECT t.value:id::bigint txn_id, to_date(t.value:date::string) txn_date, t.value:description::string description,
             t.value:person.id::bigint player_id, t.value:toTeam.id::int team_id
      FROM bronze_mlb_transactions, LATERAL variant_explode(payload:transactions) t
      WHERE t.value:typeCode::string = 'SC' AND lower(t.value:description::string) LIKE '%injured list%'""")


def silver(name, src, keys, comment, *cols):
    @dp.table(name=name, comment=comment)
    def t():
        return spark.readStream.table(src).selectExpr(*cols).dropDuplicates(keys)


silver("silver_trainer_notes", "bronze_club_trainer_notes", ["note_id"], "Trainer/coach notes (medical; masked by governance)",
       "note_id", "athlete_id", "to_timestamp(note_ts) note_ts", "author_role", "note_text")
silver("silver_il_events", "bronze_club_medical", ["event_id"], "Club IL placements/activations (medical)",
       "event_id", "athlete_id", "to_date(event_date) event_date", "action", "description")
silver("silver_wellness", "bronze_club_wellness", ["athlete_id", "survey_date"], "Daily wellness survey, one per athlete-day",
       "athlete_id", "to_date(survey_date) survey_date", "CAST(arm_soreness_0_10 AS INT) arm_soreness_0_10",
       "CAST(fatigue_0_10 AS INT) fatigue_0_10", "CAST(sleep_hours AS DOUBLE) sleep_hours")


@dp.materialized_view(comment="Club pitcher roster (pseudonymous athlete_id)")
def silver_roster():
    return spark.read.table("bronze_club_roster").select("athlete_id", "display_name", "throws").distinct()
