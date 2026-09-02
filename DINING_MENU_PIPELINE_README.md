# Dining Menu Pipeline

## Architecture

Dining menu data is generated in `routes/events.py`.

Flow:

1. Fetch UC Merced dining menu items from `https://widget.api.eagle.bigzpoon.com/menuitems`.
2. Build `PAV_nested_content` and `YWDC_nested_content` as mobile `nested_content`.
3. Replace only the generated menu portion of `routes/polygons.json`.
4. Preserve non-menu nested content such as Amenities or Hours & Info.
5. `GET /contentAPIURL` reads `routes/polygons.json` on each request and includes those dining records in `pages`.

The menu pipeline does not hardcode food items. It hardcodes dining source IDs, category IDs, and day group IDs required by the upstream dining widget API.

## Dining Locations

`routes/polygons.json` records updated by location ID:

- Pavilion: `location_id` 774, generated key `PAV_nested_content`, source location ID `61df4a34d5507a00103ee41e`
- Yablokoff-Wallace Dining Center: `location_id` 1130, generated key `YWDC_nested_content`, source location ID `628672b52903a50010fa751e`

Expected generated structure:

```json
{
  "nested_content": [
    {
      "title": "Sunday",
      "tabs": [
        {
          "title": "Breakfast",
          "sections": [
            { "header": "Menu item", "bullets": ["Description: ..."] }
          ]
        }
      ]
    }
  ]
}
```

## Schedule

The scheduler is APScheduler `BackgroundScheduler`, configured with `America/Los_Angeles`.

- Dining menu regeneration: every Sunday at 7:00 AM Pacific
- Presence pages cache refresh: every day at 7:05 AM Pacific

Only one scheduler is started per Flask process. If Render runs more than one production instance, enable `RUN_CONTENT_JOBS=true` on only one instance or move the job to a separate worker/cron service.

## Environment Variables

- `RUN_STARTUP_CONTENT_PIPELINE`: when true, starts a one-time background startup job for menus plus Presence page cache.
- `RUN_CONTENT_JOBS`: when true, starts APScheduler jobs.
- `MENU_REQUEST_DELAY_SECONDS`: delay between dining widget requests. Default: `1`.
- `MONGODB_URI`: required by `main.py` for the Flask app to start because event routes use MongoDB.
- `OPENAI_API_KEY`: currently required by `main.py` because the OpenAI client is created during app startup. Dining menu generation itself does not call OpenAI.

## Render Production Settings

Recommended for the single Render web service deployment:

```env
RUN_STARTUP_CONTENT_PIPELINE=true
RUN_CONTENT_JOBS=true
MENU_REQUEST_DELAY_SECONDS=1
MONGODB_URI=<secret>
OPENAI_API_KEY=<secret>
```

`RUN_STARTUP_CONTENT_PIPELINE=true` refreshes generated menu content after deploy/restart. `RUN_CONTENT_JOBS=true` enables the weekly Sunday regeneration.

Make sure Render installs `requirements.txt`; APScheduler is required for scheduled jobs.

## Manual Regeneration

From `Uc_merced_campus_event_api_backend_ios_android_update/`:

```bash
python3 tools/regenerate_dining_menus.py
```

To also refresh the Presence event pages cache:

```bash
python3 tools/regenerate_dining_menus.py --refresh-presence-cache
```

The command fetches fresh Pavilion and Yablokoff-Wallace Dining Center menu data, writes it into `routes/polygons.json`, and invalidates the in-process `/contentAPIURL` cache for that run.

## Verification

Check the content feed:

```bash
curl -sS https://<your-render-host>/contentAPIURL
```

Inspect `pages` for:

- Pavilion record with `location_id` 774 and non-empty `nested_content`
- Yablokoff-Wallace Dining Center record with `location_id` 1130 and non-empty `nested_content`
- day entries with `tabs`
- meal tabs with non-empty `sections`

Useful local smoke command:

```bash
python3 -m unittest tests/test_dining_menu_pipeline.py
```

## Cache Troubleshooting

`/contentAPIURL` is assembled from three sources:

- Presence event pages from a 6-hour in-memory/file cache
- dining/location content from `routes/polygons.json`
- user-generated pages from `pages.json`

Dining menus are not served from a separate response cache; `/contentAPIURL` reads `routes/polygons.json` on each request. After successful dining regeneration, the in-process content feed cache is invalidated. On production redeploy, `RUN_STARTUP_CONTENT_PIPELINE=true` refreshes generated menu content and the Presence pages cache.

Future generated cache files matching `routes/presence_*_cache.json` should not be committed.

## Common Failures

- Scheduler not running: verify `RUN_CONTENT_JOBS=true` and APScheduler is installed.
- Startup did not refresh menu: verify `RUN_STARTUP_CONTENT_PIPELINE=true` and look for startup pipeline logs.
- Missing `MONGODB_URI`: `main.py` fails during app startup.
- Missing `OPENAI_API_KEY`: the current app startup fails while creating the OpenAI client; dining generation does not use it when run directly through the manual script.
- Menu source unavailable: generation fails, logs the failing source/day/meal, and keeps the last valid `polygons.json`.
- Generated menu not injected: verify `routes/polygons.json` contains records with `location_id` 774 and 1130.
- Stale `/contentAPIURL`: run the manual regeneration command, then verify the two dining records in `/contentAPIURL`.

## Useful Logs

Look for:

- `[content_jobs] startup refresh enabled; launching startup pipeline`
- `[content_jobs] scheduler started - food menus: Sunday 07:00 Pacific`
- `[content_jobs] job started: Sunday food-menu generation`
- `[food_menu] generated menu content`
- `[food_menu] polygons.json updated successfully`
- `[content_jobs] /contentAPIURL cache invalidated`
- `[content_api] rebuilt /contentAPIURL`
- `[food_menu] generated menu rejected; existing polygons retained`
