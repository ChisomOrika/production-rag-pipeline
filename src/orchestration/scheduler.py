"""Pipeline orchestration: scheduled freshness checks and re-ingestion.

Uses APScheduler for lightweight cron-style scheduling without needing
a full Airflow deployment. For production at scale, replace with Airflow DAGs.
"""

import logging
from datetime import datetime, timezone

from apscheduler.schedulers.background import BackgroundScheduler
from sqlalchemy.orm import Session

from config.settings import settings
from src.db.session import SessionLocal
from src.freshness.tracker import FreshnessTracker
from src.ingestion.pipeline import IngestionPipeline

logger = logging.getLogger(__name__)


def run_freshness_check():
    """Scheduled job: run freshness check and reconciliation."""
    logger.info("Running scheduled freshness check")
    db = SessionLocal()
    try:
        pipeline = IngestionPipeline(db=db)
        run = pipeline.run_freshness_check()
        logger.info(f"Freshness check completed: run_id={run.id}, stale={run.stale_detected}")
    except Exception as e:
        logger.error(f"Scheduled freshness check failed: {e}")
    finally:
        db.close()


def run_source_refresh(source_urls: list[str] | None = None):
    """Scheduled job: re-check source URLs for updates and re-ingest if changed."""
    logger.info("Running scheduled source refresh")
    db = SessionLocal()
    try:
        if source_urls:
            pipeline = IngestionPipeline(db=db)
            run = pipeline.ingest_from_urls(source_urls)
            logger.info(
                f"Source refresh completed: run_id={run.id}, "
                f"processed={run.documents_processed}, stale={run.stale_detected}"
            )
        else:
            logger.info("No source URLs configured for refresh")
    except Exception as e:
        logger.error(f"Scheduled source refresh failed: {e}")
    finally:
        db.close()


class PipelineScheduler:
    """Manages scheduled pipeline jobs."""

    def __init__(self):
        self.scheduler = BackgroundScheduler()
        self._source_urls: list[str] = []

    def configure(self, source_urls: list[str] | None = None):
        """Configure the scheduler with source URLs to monitor."""
        self._source_urls = source_urls or []

    def start(self):
        """Start the scheduler with configured jobs."""
        # Freshness check every N hours
        self.scheduler.add_job(
            run_freshness_check,
            "interval",
            hours=settings.staleness_check_interval_hours,
            id="freshness_check",
            replace_existing=True,
        )

        # Source refresh every 24 hours
        if self._source_urls:
            self.scheduler.add_job(
                run_source_refresh,
                "interval",
                hours=24,
                args=[self._source_urls],
                id="source_refresh",
                replace_existing=True,
            )

        self.scheduler.start()
        logger.info(
            f"Scheduler started: freshness check every {settings.staleness_check_interval_hours}h"
        )

    def stop(self):
        """Stop the scheduler."""
        self.scheduler.shutdown()
        logger.info("Scheduler stopped")

    def trigger_freshness_check(self):
        """Manually trigger a freshness check."""
        run_freshness_check()

    def trigger_source_refresh(self, urls: list[str] | None = None):
        """Manually trigger source refresh."""
        run_source_refresh(urls or self._source_urls)

    def get_jobs(self) -> list[dict]:
        """List scheduled jobs."""
        return [
            {
                "id": job.id,
                "next_run_time": str(job.next_run_time),
                "trigger": str(job.trigger),
            }
            for job in self.scheduler.get_jobs()
        ]
