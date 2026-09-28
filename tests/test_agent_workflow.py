from src.agent.controller import AgentController
from src.agent.query_types import AnalysisRequest
from src.agent.workflow import summarize, deterministic_report
def test_blocked_workflow_summary_and_report_preserve_fallback():
 r=AgentController().analyze(AnalysisRequest('Where is the river?',({'id':'x','modality':'rgb'},)))
 assert summarize(r)['execution']['fallback']=='NONE'
 assert deterministic_report(r)['audit']['capability']=='BLOCKED'
