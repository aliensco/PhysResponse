from physedit.native_observation_logic import infer

def test_missing_numeric_conjunct_never_becomes_positive():
    p=dict(contacts=['contact'],complete=False)
    assert infer(p,dict(measurability='measurable',events={'contact':'yes'}))[0] is None
    assert infer(p,dict(measurability='measurable',events={'contact':'no'}))[0]==0
    assert infer(p,dict(measurability='measurable',events={'contact':'uncertain'}))[0] is None


def test_complete_chain_requires_all_contacts_and_order():
    p=dict(contacts=['a','b'],order='order',complete=True)
    h=dict(measurability='measurable',events={'a':'yes','b':'yes','order':'yes'})
    assert infer(p,h)[0]==1
    h['events']['order']='no';assert infer(p,h)[0]==0
    h['events']['order']='uncertain';assert infer(p,h)[0] is None
    h['events']['a']='no';assert infer(p,h)[0]==0
    h['measurability']='unmeasurable';assert infer(p,h)[0] is None


def test_order_cannot_refute_if_prerequisite_unknown():
    p=dict(contacts=['a','b'],order='order',complete=True)
    assert infer(p,dict(measurability='measurable',events={'a':'yes','b':'uncertain','order':'no'}))[0] is None
