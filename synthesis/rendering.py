import base64
import streamlit as st

def render_3d_objs(file_paths, size_limit_mb = 200):
    objects_data = []
    total_size_bytes = 0
    size_limit_bytes = size_limit_mb * 1024 * 1024

    for path in file_paths:
        with open(path, "rb") as f:
            file_data = f.read()

        total_size_bytes += len(file_data)

        if total_size_bytes > size_limit_bytes:
            st.warning(
                f"Total file size exceeds the {size_limit_mb}MB limit. 3D view will not be rendered to prevent performance issues.")
            return

        b64 = base64.b64encode(f.read()).decode("utf-8")

        name = path.name.lower()
        if "crown" in name:
            color = "0x4CAF50"
        elif "trunk" in name:
            color = "0x8B4513"
        elif "building" in name or "bag" in name:
            color = "0x9E9E9E"
        else:
            color = "0x4CAF50"

        objects_data.append(f"{{ data: '{b64}', color: {color} }}")

    js_array = "[\n" + ",\n".join(objects_data) + "\n]"

    html_code = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>body {{ margin: 0; overflow: hidden; }} canvas {{ display: block; }}</style>
        </head>
        <body>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/OBJLoader.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
            <script>
                var scene = new THREE.Scene();
                scene.background = new THREE.Color(0xf0f4f8);

                var camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 5000);
                var renderer = new THREE.WebGLRenderer({{ antialias: true }});
                renderer.setSize(window.innerWidth, window.innerHeight);
                document.body.appendChild(renderer.domElement);

                var controls = new THREE.OrbitControls(camera, renderer.domElement);
                controls.enableDamping = true;
                controls.target.set(0,0,0);
                
                var ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
                scene.add(ambientLight);
                var dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
                dirLight.position.set(100, 200, 50);
                scene.add(dirLight);

                var loader = new THREE.OBJLoader();
                var objectsToLoad = {js_array};
                var loadedCount = 0;
                var mainGroup = new THREE.Group();
                scene.add(mainGroup);

                objectsToLoad.forEach(function(item) {{
                    var objData = atob(item.data);
                    var blob = new Blob([objData], {{ type: 'text/plain' }});
                    var url = URL.createObjectURL(blob);

                    loader.load(url, function (object) {{
                        object.traverse(function(child) {{
                            if (child.isMesh) {{
                                child.material = new THREE.MeshLambertMaterial({{ color: item.color, side: THREE.DoubleSide }});
                            }}
                        }});

                        mainGroup.add(object);
                        loadedCount++;

                        // Once all files are loaded, center the camera
                        if (loadedCount === objectsToLoad.length) {{
                            var box = new THREE.Box3().setFromObject(mainGroup);
                            var center = box.getCenter(new THREE.Vector3());
                            mainGroup.position.sub(center); 

                            var size = box.getSize(new THREE.Vector3()).length();
                            camera.position.z = size * 1.2;
                            camera.position.y = size * 0.5;
                            camera.lookAt(0, 0, 0);
                        }}
                    }});
                }});

                function animate() {{
                    requestAnimationFrame(animate);
                    controls.update();
                    renderer.render(scene, camera);
                }}
                animate();
            </script>
        </body>
        </html>
        """
    st.iframe(html_code, height=600)