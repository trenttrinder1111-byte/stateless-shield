import os
import shutil
import pytest
import numpy as np
from shield import StatelessShield

@pytest.fixture
def clean_test_dir():
    test_dir = "test_sandbox"
    os.makedirs(test_dir, exist_ok=True)
    yield test_dir
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

def test_shield_add_and_search():
    shield = StatelessShield(vector_dim=4)
    vector = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    shield.add(node_id=1, vector=vector, payload="test payload")
    
    results = shield.search(vector, top_k=1)
    assert len(results) == 1
    assert results[0][0] == 1
    assert results[0][1] == "test payload"

def test_shield_save_and_load(clean_test_dir):
    prefix = os.path.join(clean_test_dir, "my_shield")

    shield = StatelessShield(vector_dim=4)
    vector = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    shield.add(node_id=42, vector=vector, payload="persistence test")
    
    shield.save(prefix)
    
    assert os.path.exists(f"{prefix}.index")
    assert os.path.exists(f"{prefix}.json")

    shield_loaded = StatelessShield(vector_dim=4)
    shield_loaded.load(prefix)
    
    results = shield_loaded.search(vector, top_k=1)
    assert len(results) == 1
    assert results[0][0] == 42
    assert results[0][1] == "persistence test"

def test_path_traversal_protection():
    shield = StatelessShield(vector_dim=4)
    
    with pytest.raises(ValueError, match="Path traversal attempt blocked"):
        shield.save("../outside_sandbox")

    with pytest.raises(ValueError, match="Path traversal attempt blocked"):
        shield.load("/etc/passwd")
