#include "framework.h"

//TODO:
//nyilatkozat
//kommentek tï¿½rlï¿½se
//forrï¿½smegjelï¿½lï¿½sek

//Szirmay-Kalos Lï¿½szlï¿½ munkï¿½ssï¿½ga alapjï¿½n
struct Material {
	vec3 ka, kd, ks;
	float shininess;
	Material(vec3 _kd, vec3 _ks, float _shininess) : ka(_kd* M_PI), kd(_kd), ks(_ks) { shininess = _shininess; }
};

//Szirmay-Kalos Lï¿½szlï¿½ munkï¿½ssï¿½ga alapjï¿½n
struct Hit {
	float t;
	vec3 position, normal;
	Material* material;
	Hit() { t = -1; }
};

//Szirmay-Kalos Lï¿½szlï¿½ munkï¿½ssï¿½ga alapjï¿½n
struct Ray {
	vec3 start, dir;
	Ray(vec3 _start, vec3 _dir) : start(_start), dir(_dir) {}
};

//Szirmay-Kalos Lï¿½szlï¿½ munkï¿½ssï¿½ga alapjï¿½n
class Intersectable {
protected:
	Material* material;
public:
	virtual Hit intersect(const Ray& ray) = 0;
};

//Szirmay-Kalos Lï¿½szlï¿½ munkï¿½ssï¿½ga alapjï¿½n
class Camera {
	vec3 eye, lookat, right, up;
	float fov;
public:
	void set(vec3 _eye, vec3 _lookat, vec3 vup, float _fov) {
		eye = _eye; lookat = _lookat, fov = _fov;
		vec3 w = eye - lookat;
		float windowsSize = length(w) * tanf(fov/2);
		right = normalize(cross(vup, w)) * windowsSize;
		up = normalize(cross(w, right)) * windowsSize;
	}

	Ray getRay(int X, int Y) {
		vec3 dir = lookat + right * (2 * (X + 0.5f) / windowWidth - 1) + up * (2 * (Y + 0.5f) / windowHeight - 1) - eye;
		return Ray(eye, dir);
	}

	/*void Animate(float dt) {
		vec3 d = eye - lookat;
		eye = vec3(d.x * cos(dt) + d.z )
	}*/
};

struct Light {
	vec3 direction, Le; //irï¿½ny, intenzitï¿½s
	Light(vec3 _direction, vec3 _Le) {
		direction = normalize(_direction);
		Le = _Le;
	}
};

const float epsilon = 0.0001f;

//TEST
struct Sphere : public Intersectable {
	vec3 center;
	float radius;

	Sphere(const vec3& _center, float _radius, Material* _material) {
		center = _center;
		radius = _radius;
		material = _material;
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		vec3 dist = ray.start - center;
		float a = dot(ray.dir, ray.dir);
		float b = dot(dist, ray.dir) * 2.0f;
		float c = dot(dist, dist) - radius * radius;
		float discr = b * b - 4.0f * a * c;
		if (discr < 0) return hit;
		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;	// t1 >= t2 for sure
		float t2 = (-b - sqrt_discr) / 2.0f / a;
		if (t1 <= 0) return hit;
		hit.t = (t2 > 0) ? t2 : t1;
		hit.position = ray.start + ray.dir * hit.t;
		hit.normal = (hit.position - center) * (1.0f / radius);
		hit.material = material;
		return hit;
	}
};

class Triangle : public Intersectable{
	vec3 A, B, C;
	vec3 normal;
	Material* material;
public:
	Triangle(std::vector<vec3> _points, Material* _material){ //4 points
		A = _points[0];
		B = _points[1];
		C = _points[2];
		normal = normalize(cross(B - A, C - A));
		material = _material;
	}
	Hit intersect(const Ray& ray) {
		Hit hit;
		//TODO: ha hit, akkor n felénk néz-e? Ha igen, skip
		float t = dot((A - ray.start), normal) / dot(ray.dir, normal);
		if (t < 0) return hit;
		vec3 p = ray.start + t * ray.dir;

		if (dot(cross(B-A,p-A),normal) <= 0) return hit;
		if (dot(cross(C-B,p-B),normal) <= 0) return hit;
		if (dot(cross(A-C,p-C),normal) <= 0) return hit;
		
		hit.normal = normal;
		hit.position = p;
		hit.material = material;
		hit.t = t;
		return hit;
	}
};

/*class Square : public Intersectable {
	Triangle 
};*/

class Scene {
	std::vector<Intersectable*> objects;
	std::vector<Light*> lights;
	Camera camera;
	vec3 La; //ambiens fï¿½ny
public:
	void build() {
		vec3 eye = vec3(1.8f, 0.5f, 1.8f), vup = vec3(0, 1, 0), lookat = vec3(0,0.5,0);
		float fov = 45 * M_PI / 180;
		camera.set(eye, lookat, vup, fov);

		La = vec3(0.0f, 0.0f, 0.0f);
		vec3 lightDirection(1,0.3,1), Le(2,2,2);
		lights.push_back(new Light(lightDirection, Le));
		

		//2 material kï¿½szï¿½tï¿½se
		vec3 kd1(0.3f, 0.3f, 0.3f), ks(2, 2, 2); //diffï¿½z ï¿½s spekulï¿½ris visszaverï¿½dï¿½si tï¿½nyezï¿½k (kell?)
		Material* material = new Material(kd1, ks, 50); //50?

		vec3 A(vec3(0.0,  0.0,  0.0)),	//1
		B(vec3(0.0,  0.0,  1.0)),		//2
		C(vec3(0.0,  1.0,  0.0)),		//3
		D(vec3(0.0,  1.0,  1.0)),		//4
		E(vec3(1.0,  0.0,  0.0)),		//5
		F(vec3(1.0,  0.0,  1.0)),		//6
		G(vec3(1.0,  1.0,  0.0)),		//7
		H(vec3(1.0,  1.0,  1.0));		//8

		//Jobb távolabbi oldal
		std::vector<vec3> cubeTriangles = {A,G,E};
		objects.push_back(new Triangle(cubeTriangles, material));
		cubeTriangles = { A,C,G };
		objects.push_back(new Triangle(cubeTriangles, material));
		

		//Bal távolabbi oldal
		cubeTriangles = { A,D,C };
		objects.push_back(new Triangle(cubeTriangles, material));
		cubeTriangles = { A,B,D };
		objects.push_back(new Triangle(cubeTriangles, material));

		cubeTriangles = { C,H,G };
		objects.push_back(new Triangle(cubeTriangles, material));
		cubeTriangles = { C,D,H };
		objects.push_back(new Triangle(cubeTriangles, material));

		//Jobb közelebbi oldal
		/*cubeTriangles = { E,G,H };
		objects.push_back(new Triangle(cubeTriangles, material));
		cubeTriangles = { E,H,F };
		objects.push_back(new Triangle(cubeTriangles, material));*/

		//Alsó oldal
		cubeTriangles = { A,E,F };
		objects.push_back(new Triangle(cubeTriangles, material));
		cubeTriangles = { A,F,B };
		objects.push_back(new Triangle(cubeTriangles, material));

		//Bal közelebbi oldal
		/*cubeTriangles = { B,F,H };
		objects.push_back(new Triangle(cubeTriangles, material));
		cubeTriangles = { B,H,D };
		objects.push_back(new Triangle(cubeTriangles, material));*/

		kd1 = vec3(1.0f, 0.3f, 0.3f), ks = vec3(2, 2, 2);
		material = new Material(kd1, ks, 50); //50?
		objects.push_back(new Sphere(normalize(lightDirection), 0.2, material));
		
	}

	void render(std::vector<vec4>& image) { //virt. vilï¿½g lefï¿½nykï¿½pezï¿½se
		for (int Y = 0; Y < windowHeight; Y++) {
#pragma omp parallel for
			for (int X = 0; X < windowWidth; X++) {
				vec3 color = trace(camera.getRay(X, Y)); //minden pixelnek meghatï¿½rozzuk a szï¿½nï¿½t
				image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
			}
		}
	}

	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		for (Intersectable* obj : objects) {
			Hit hit = obj->intersect(ray);
			//printf("Intersect - %lf,%lf,%lf - %lf,%lf,%lf\n", hit.normal.x, hit.normal.y, hit.normal.z, hit.position.x, hit.position.y, hit.position.z);
			if (hit.t > 0 && (bestHit.t < 0 || bestHit.t > hit.t)) bestHit = hit;
		}
		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}

	bool shadowIntersect(Ray ray) {	// for directional lights
		for (Intersectable* object : objects) if (object->intersect(ray).t > 0) return true;
		return false;
	}

	vec3 trace(Ray ray, int depth = 0) {
		Hit hit = firstIntersect(ray);
		if (hit.t < 0) return La;
		
		vec3 outRadiance = hit.material->ka * La;
		for (Light* light : lights) {
			Ray shadowRay(hit.position + hit.normal * epsilon, light->direction);
			float cosTheta = dot(hit.normal, light->direction);
			if (cosTheta > 0 && !shadowIntersect(shadowRay)) {	// shadow computation
				outRadiance = outRadiance + light->Le * hit.material->kd * cosTheta;
				vec3 halfway = normalize(-ray.dir + light->direction);
				float cosDelta = dot(hit.normal, halfway);
				if (cosDelta > 0) outRadiance = outRadiance + light->Le * hit.material->ks * powf(cosDelta, hit.material->shininess);
			}
		}
		return outRadiance;
	}
};

GPUProgram gpuProgram;
Scene scene;

const char *vertexSource = R"(
	#version 330
    precision highp float;

	layout(location = 0) in vec2 cVertexPosition;	// Attrib Array 0
	out vec2 texcoord;

	void main() {
		texcoord = (cVertexPosition + vec2(1, 1))/2;							// -1,1 to 0,1
		gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1); 		// transform to clipping space
	}
)";

const char *fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	in  vec2 texcoord;			// interpolated texture coordinates
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = texture(textureUnit, texcoord); 
	}
)";

class FullScreenTexturedQuad {
	unsigned int vao;	// vertex array object id and texture id
	Texture texture;
public:
	FullScreenTexturedQuad(int windowWidth, int windowHeight, std::vector<vec4>& image)
		: texture(windowWidth, windowHeight, image) 
	{
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo;		// vertex buffer objects
		glGenBuffers(1, &vbo);	// Generate 1 vertex buffer objects

		// vertex coordinates: vbo0 -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
		float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };	// two triangles forming a quad
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified 
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed
	}

	void Draw() {
		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		gpuProgram.setUniform(texture, "textureUnit");
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);	// draw two triangles forming a quad
	}
};

FullScreenTexturedQuad * fullScreenTexturedQuad;

void onInitialization(){
	glViewport(0, 0, windowWidth, windowHeight);
	scene.build();
	std::vector<vec4> image(windowWidth * windowHeight);
	scene.render(image);
	fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);
	gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}

void onDisplay(){
	fullScreenTexturedQuad->Draw();
	glutSwapBuffers();
}

void onMouse(int button, int state, int pX, int pY){
	//TODO
}
void onKeyboard(unsigned char key, int pX, int pY){}
void onKeyboardUp(unsigned char key, int pX, int pY){}
void onMouseMotion(int pX, int pY){}
void onIdle(){}