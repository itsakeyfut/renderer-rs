//! Integration tests for model loading.

use std::path::Path;

use renderer_resources::Model;

#[test]
fn test_load_gltf_model() {
    // Path to the test glTF file
    let model_path = Path::new("../../assets/models/a_contortionist_dancer/scene.gltf");

    // Skip test if file doesn't exist (CI environment may not have assets)
    if !model_path.exists() {
        println!("Skipping test: model file not found at {:?}", model_path);
        return;
    }

    // Load the model
    let model = Model::load(model_path).expect("Failed to load glTF model");

    // Verify the model was loaded correctly
    assert!(
        !model.meshes.is_empty(),
        "Model should have at least one mesh"
    );

    // Verify vertex data was extracted
    for (i, mesh) in model.meshes.iter().enumerate() {
        assert!(
            !mesh.positions.is_empty(),
            "Mesh {} should have positions",
            i
        );
        assert_eq!(
            mesh.normals.len(),
            mesh.positions.len(),
            "Mesh {} should have same number of normals as positions",
            i
        );
        assert_eq!(
            mesh.tex_coords.len(),
            mesh.positions.len(),
            "Mesh {} should have same number of tex coords as positions",
            i
        );
        assert_eq!(
            mesh.tangents.len(),
            mesh.positions.len(),
            "Mesh {} should have same number of tangents as positions",
            i
        );
        assert!(!mesh.indices.is_empty(), "Mesh {} should have indices", i);
    }

    // Verify AABB was computed
    assert!(
        model.aabb_min.x < model.aabb_max.x,
        "AABB min x should be less than max x"
    );
    assert!(
        model.aabb_min.y < model.aabb_max.y,
        "AABB min y should be less than max y"
    );
    assert!(
        model.aabb_min.z < model.aabb_max.z,
        "AABB min z should be less than max z"
    );

    // Print model stats
    println!("Loaded model with {} meshes", model.meshes.len());
    println!("Total vertices: {}", model.total_vertex_count());
    println!("Total triangles: {}", model.total_triangle_count());
    println!("Materials: {}", model.materials.len());
    println!(
        "AABB: [{:.2}, {:.2}, {:.2}] to [{:.2}, {:.2}, {:.2}]",
        model.aabb_min.x,
        model.aabb_min.y,
        model.aabb_min.z,
        model.aabb_max.x,
        model.aabb_max.y,
        model.aabb_max.z
    );
}
