"""
Webhook endpoints for content ingestion from Sanity CMS.
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks, Header
from pydantic import BaseModel
from typing import Optional, Dict, Any
import hmac
import hashlib
import os
from datetime import datetime

from src.api.ingestion_service import ContentIngestionService

router = APIRouter(prefix="/webhooks", tags=["webhooks"])

# Initialize ingestion service
ingestion_service = ContentIngestionService()


class SanityWebhookPayload(BaseModel):
    """Sanity webhook payload structure."""
    _id: str
    _type: str
    _rev: Optional[str] = None
    _createdAt: Optional[str] = None
    _updatedAt: Optional[str] = None
    title: Optional[str] = None


def verify_sanity_signature(payload: bytes, signature: str, secret: str) -> bool:
    """
    Verify Sanity webhook signature.
    
    Args:
        payload: Raw request body
        signature: Signature from Sanity-Webhook-Signature header
        secret: Webhook secret from environment
    
    Returns:
        True if signature is valid
    """
    if not secret:
        # If no secret configured, skip verification (dev mode)
        return True
    
    expected_signature = hmac.new(
        secret.encode(),
        payload,
        hashlib.sha256
    ).hexdigest()
    
    return hmac.compare_digest(signature, expected_signature)


@router.post("/sanity/article-published")
async def sanity_article_published(
    payload: SanityWebhookPayload,
    background_tasks: BackgroundTasks,
    sanity_webhook_signature: Optional[str] = Header(None)
):
    """
    Webhook endpoint for Sanity CMS article published/updated events.
    
    Sanity will call this endpoint when:
    - A new article is published
    - An existing article is updated
    
    The article will be fetched, chunked, and embedded in the background.
    """
    # Verify webhook signature (if configured)
    webhook_secret = os.getenv('SANITY_WEBHOOK_SECRET')
    if webhook_secret and sanity_webhook_signature:
        # Note: For proper signature verification, we'd need the raw body
        # This is a simplified version
        pass
    
    print(f"[*] Webhook received for article: {payload._id}")
    print(f"    Type: {payload._type}")
    print(f"    Title: {payload.title}")
    
    # Validate payload
    if payload._type != "articles":
        raise HTTPException(
            status_code=400,
            detail=f"Invalid document type: {payload._type}. Expected 'articles'"
        )
    
    # Queue ingestion in background
    background_tasks.add_task(
        ingestion_service.ingest_article_by_id,
        article_id=payload._id
    )
    
    return {
        "status": "accepted",
        "message": f"Article {payload._id} queued for ingestion",
        "article_id": payload._id,
        "title": payload.title
    }


@router.post("/sanity/sync-all")
async def sync_all_articles(
    background_tasks: BackgroundTasks,
    limit: Optional[int] = None
):
    """
    Manually trigger sync of all articles from Sanity.
    
    Args:
        limit: Optional limit on number of articles to sync
    
    This endpoint allows manual full sync of content.
    """
    print(f"[*] Manual sync triggered (limit: {limit or 'all'})")
    
    # Queue full sync in background
    background_tasks.add_task(
        ingestion_service.sync_all_articles,
        limit=limit
    )
    
    return {
        "status": "accepted",
        "message": f"Full sync queued (limit: {limit or 'all'})"
    }


@router.get("/sanity/status")
async def get_sync_status():
    """
    Get current sync status and statistics.
    """
    stats = ingestion_service.get_stats()
    
    return {
        "status": "healthy",
        "statistics": stats
    }
