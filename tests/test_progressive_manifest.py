import os
from unittest.mock import MagicMock, patch

from src.lib.crawler.cleaned import cleaned_sections
from src.adapters.interfaces.crawler import CrawledPage
from src.core.saving.ingest_manifest import create_manifest, manifest_snapshot, verify_control


def test_headings_without_boundaries_do_not_create_fake_sections():
    parts = cleaned_sections({"content":{"headings":["A","B"],"core":"Hello 🧠"},"facts":{"contacts":{"emails":["hi@example.com"]},"ctas":["Join"]}})
    assert [part.title for part in parts] == ["Main content", "Contacts", "Calls to action"]
    assert len(parts[0].text) == 7


def test_explicit_sections_preserve_heading_paths():
    parts=cleaned_sections({"content":{"sections":[{"heading":"Install","text":"Run setup","heading_path":["Docs","Install"]}]}})
    assert parts[0].heading_path == ["Docs","Install"]
    assert parts[0].text == "Run setup"


def test_manifest_deduplicates_canonical_pages_and_preserves_skips():
    root=MagicMock();root.find_one.return_value=None
    sections=MagicMock()
    pages=[CrawledPage(url="https://a.test",text="Hello 🧠"),CrawledPage(url="https://a.test",text="duplicate"),CrawledPage(url="https://a.test/no",text="",skip_reason="Unreadable")]
    with patch("src.core.saving.ingest_manifest.collection",return_value=root),patch("src.core.saving.ingest_manifest.sections_collection",return_value=sections):
        create_manifest("brain","import",pages,{})
    assert sections.update_one.call_count == 1
    doc=sections.update_one.call_args.args[1]["$setOnInsert"]
    assert doc["chars"] == 7
    manifest=root.update_one.call_args.args[1]["$setOnInsert"]
    assert len(manifest["pages"]) == 2
    assert manifest["pages"][1]["reason"] == "Unreadable"


def test_customer_snapshot_omits_source_and_costs_and_keeps_active_failures():
    root=MagicMock();root.find_one.return_value={"pages":[]}
    sections=MagicMock();sections.find.return_value.sort.return_value=[
        {"_id":"s1","status":"completed","chars":3,"text":"private","cost":{"tokens":99}},
        {"_id":"s2","status":"failed","chars":7,"error":"secret raw provider error"},
        {"_id":"s3","status":"pending","chars":9},
    ]
    with patch("src.core.saving.ingest_manifest.collection",return_value=root),patch("src.core.saving.ingest_manifest.sections_collection",return_value=sections):
        result=manifest_snapshot("brain","import")
    assert result["status"] == "started"
    assert result["characters_completed"] == 3
    assert result["progress"] == {"done":2,"total":3,"succeeded":1,"failed":1}
    assert "text" not in result["sections"][0]
    assert "cost" not in result["sections"][0]
    assert "secret" not in str(result)


def test_control_signature_is_bound_to_brain_action_import_and_expiry():
    import hashlib,hmac,time
    stamp=str(int(time.time()))
    signature=hmac.new(b"test",f"ingest-control:brain:import:dispatch:{stamp}".encode(),hashlib.sha256).hexdigest()
    token=f"{stamp}.{signature}"
    with patch.dict(os.environ,{"USAGE_CONTEXT_SECRET":"test"}):
        assert verify_control("brain","import","dispatch",token)
        assert not verify_control("other","import","dispatch",token)
        assert not verify_control("brain","import","retry",token)
        assert not verify_control("brain","import","dispatch","0."+signature)


def test_platform_envelope_does_not_depend_on_gateway_usage_header():
    import base64,hashlib,hmac,json,time
    from types import SimpleNamespace
    from fastapi import HTTPException
    from src.core.saving.ingest_manifest import pipeline_context
    body={"brain_id":"brain","job_id":"job","exp":int(time.time())+60,"configuration":{"version":"v1","routes":[]}}
    encoded=base64.urlsafe_b64encode(json.dumps(body).encode()).decode().rstrip("=")
    signature=hmac.new(b"test",f"pipeline-v1.{encoded}".encode(),hashlib.sha256).hexdigest()
    request=SimpleNamespace(headers={"X-Pipeline-Context":f"pipeline-v1.{encoded}.{signature}"})
    with patch.dict(os.environ,{"USAGE_CONTEXT_SECRET":"test"}):
        assert pipeline_context(request,"brain","task")["configuration"]["version"]=="v1"
        try:
            pipeline_context(request,"other-brain","task")
            assert False,"A different brain must not accept this envelope"
        except HTTPException as error:
            assert error.status_code==403


def test_routing_is_operation_scoped_and_does_not_change_fallbacks():
    from src.core.saving.pipeline_runtime import bind_operation,reset_operation,routed_client,_stage
    fallback=object();selected=object()
    operation=bind_operation({"configuration":{"routes":[{"stage":"ingestion.scout","provider":"amazon_bedrock","model":"selected"}]}})
    stage=_stage.set("scout")
    try:
        with patch("src.core.saving.pipeline_runtime.route_client",return_value=selected):
            assert routed_client(fallback) is selected
            _stage.set("architect")
            assert routed_client(fallback) is fallback
    finally:
        _stage.reset(stage)
        reset_operation(operation)
    assert routed_client(fallback) is fallback


def test_url_worker_waits_for_cleaned_manifest_without_early_ingestion():
    from types import SimpleNamespace
    from src.workers.tasks import ingestion
    events=[]
    crawler=MagicMock()
    def crawl(request):
        events.append("crawl-and-cleanup-finished")
        return [CrawledPage(url="https://example.com",text="Body")]
    crawler.crawl_pages.side_effect=crawl
    def manifest(*args,**kwargs):
        assert events==["crawl-and-cleanup-finished"]
        events.append("manifest-ready")
    with patch("src.core.instances.crawler_adapter",crawler),patch("src.core.saving.ingest_manifest.create_manifest",side_effect=manifest),patch.object(ingestion,"set_ingestion_task_status"),patch.object(ingestion.ingest_data,"apply_async") as queue:
        ingestion.ingest_url.run.__func__(SimpleNamespace(request=SimpleNamespace(id="import")),{"brain_id":"brain","url":"https://example.com","progressive":True})
        queue.assert_not_called()
    assert events==["crawl-and-cleanup-finished","manifest-ready"]


def test_pinned_defaults_preserve_small_and_large_roles_after_vm_model_changes():
    from src.core.saving.pipeline_runtime import bind_operation,reset_operation,routed_client
    token=bind_operation({"configuration":{"defaults":{"small":{"provider":"amazon_bedrock","model":"old-small"},"large":{"provider":"amazon_bedrock","model":"old-large"}}}})
    try:
        with patch("src.core.saving.pipeline_runtime.configured_model",return_value=("amazon_bedrock","new-model")),patch("src.core.saving.pipeline_runtime.route_client",side_effect=lambda provider,model,region:model):
            assert routed_client(object(),"small")=="old-small"
            assert routed_client(object(),"large")=="old-large"
    finally:
        reset_operation(token)


def test_pages_with_missing_cleanup_remain_visible_as_skipped():
    from src.lib.crawler.client_lumen import LumenCrawlerClient
    page=LumenCrawlerClient()._to_crawled_page({"ok":True,"url":"https://example.com/empty"})
    assert page.url=="https://example.com/empty"
    assert page.skip_reason=="Cleaned content is unavailable"
    assert page.text==""
