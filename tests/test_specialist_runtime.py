from src.specialist_runtime import SpecialistRuntime

class Fake:
    def __init__(self): self.closed=False
    def close(self): self.closed=True

def test_same_route_is_constructed_once():
    runtime=SpecialistRuntime(); made=[]
    def factory(): made.append(Fake()); return made[-1]
    with runtime.use("OPTICAL_SAR_ANALYSIS",factory) as first: pass
    with runtime.use("OPTICAL_SAR_ANALYSIS",factory) as second: pass
    assert first is second and runtime.status()["construction_counts"]["OPTICAL_SAR_ANALYSIS"]==1

def test_route_switch_evicts_previous_controller():
    runtime=SpecialistRuntime(); first=Fake()
    with runtime.use("SINGLE_IMAGE_VQA",lambda:first): pass
    with runtime.use("TEMPORAL_CHANGE_DESCRIPTION",Fake): pass
    assert first.closed and runtime.status()["resident_specialist"]=="TEMPORAL_CHANGE_DESCRIPTION"

def test_release_all_clears_runtime_state():
    runtime=SpecialistRuntime(); item=Fake()
    with runtime.use("SINGLE_IMAGE_SCENE_DESCRIPTION",lambda:item): pass
    runtime.release_all()
    assert item.closed and runtime.status()["loaded_routes"]==[]
