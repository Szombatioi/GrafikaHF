#include "framework.h"
#include <algorithm>
bool octa = false, cube = true;
//TODO:
//nyilatkozat
//kommentek t�rl�se
//forr�smegjel�l�sek

//Szirmay-Kalos L�szl� munk�ss�ga alapj�n
struct Material {
	vec3 ka, kd, ks;
	float shininess;
	Material(vec3 _kd, vec3 _ks, float _shininess) : ka(_kd* M_PI), kd(_kd), ks(_ks) { shininess = _shininess; }
};

//Szirmay-Kalos L�szl� munk�ss�ga alapj�n
struct Hit {
	float t;
	vec3 position, normal;
	Material* material;
	Hit() { t = -1; }
};

//Szirmay-Kalos L�szl� munk�ss�ga alapj�n
struct Ray {
	vec3 start, dir;
	Ray(vec3 _start, vec3 _dir) : start(_start), dir(_dir) {}
};

//Szirmay-Kalos L�szl� munk�ss�ga alapj�n
class Intersectable {
protected:
	Material* material;
public:
	virtual Hit intersect(const Ray& ray) = 0;
};

//Szirmay-Kalos L�szl� munk�ss�ga alapj�n
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
	vec3 direction, Le; //ir�ny, intenzit�s
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

	Triangle(vec3 _A, vec3 _B, vec3 _C, Material* _material) {
		std::vector<vec3> p = {_A,_B,_C};
		Triangle(p, _material);
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
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

class Square : public Intersectable {
	std::vector<Triangle> triangles; //2
public:
	Square(std::vector<Triangle> t) {
		triangles = std::vector<Triangle>();
		triangles.push_back(t[0]);
		triangles.push_back(t[1]);

	}

	Square(vec3 A, vec3 B, vec3 C, vec3 D, Material* mat) {
		triangles = std::vector<Triangle>();
		std::vector<vec3> t1 = { A, B, C };
		std::vector<vec3> t2 = { B, C, D };
		triangles.push_back(Triangle(t1, mat));
		triangles.push_back(Triangle(t2, mat));
	}

	Hit intersect(const Ray& ray) {
		Hit bestHit;
		for (Triangle t : triangles) {
			Hit hit = t.intersect(ray);
			if (hit.t > 0 && (bestHit.t < 0 || bestHit.t > hit.t)) bestHit = hit;
		}
		return bestHit;
	}
};

class Cube : public Intersectable {
	std::vector<Square> sides;
public:
	
	Cube(std::vector<Square> s) {
		sides = s;
	}

	Hit intersect(const Ray& ray) {
		Hit bestHit;
		for (Square s : sides) {
			Hit hit = s.intersect(ray);
			if (hit.t > 0 && (bestHit.t < 0 || bestHit.t > hit.t)) bestHit = hit;
		}
		return bestHit;
	}
};


//TODO: Triangles Square helyett
class Icosahedron : public Intersectable {
	std::vector<Square> sides;
public:
	Icosahedron(std::vector<Square> s) {
		sides = s;
	}
	Hit intersect(const Ray& ray) {
		Hit bestHit;
		for (Square s : sides) {
			Hit hit = s.intersect(ray);
			if (hit.t > 0 && (bestHit.t < 0 || bestHit.t > hit.t)) bestHit = hit;
		}
		return bestHit;
	}
};

class Octahedron : public Intersectable {
	std::vector<Square> sides;
public:
	Octahedron(std::vector<Square> s) {
		sides = s;
	}
	Hit intersect(const Ray& ray) {
		Hit bestHit;
		for (Square s : sides) {
			Hit hit = s.intersect(ray);
			if (hit.t > 0 && (bestHit.t < 0 || bestHit.t > hit.t)) bestHit = hit;
		}
		return bestHit;
	}
};

class Dodecahedron : public Intersectable {
	std::vector<Triangle> triangles;
	std::vector<Square> sides;
public:
	Dodecahedron(std::vector<Triangle> t) {
		triangles = t;
	}

	Hit intersect(const Ray& ray) {
		Hit bestHit;
		for (Triangle t : triangles) {
			Hit hit = t.intersect(ray);
			if (hit.t > 0 && (bestHit.t < 0 || bestHit.t > hit.t)) bestHit = hit;
		}
		return bestHit;
	}
};

class Scene {
	std::vector<Intersectable*> objects;
	std::vector<Light*> lights;
	Camera camera;
	vec3 La;
public:
	void build() {
		//vec3 eye = vec3(-4,1.5,4), vup = vec3(0, 1, 0), lookat = vec3(0,0,0); //tests
		vec3 eye = vec3(2.5f, 1.5f, 2.5f), vup = vec3(0, 1, 0), lookat = vec3(0,0.8,0);
		float fov = 45 * M_PI / 180;
		camera.set(eye, lookat, vup, fov);

		La = vec3(0.0f, 0.0f, 0.0f);
		vec3 lightDirection(2,0.5,3), Le(2,2,2);
		lights.push_back(new Light(lightDirection, Le));

		vec3 kd1(0.3f, 0.3f, 0.3f), ks(2, 2, 2);
		Material* material = new Material(kd1, ks, 50);

		vec3 A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T;
		std::vector<Square> sides;
		std::vector<Triangle> triangles;

		A = vec3(0, 0.425 , -0.265),
		B = vec3(0.425, 0.265, 0),
		C = vec3(0.425, -0.265, 0),
		D = vec3(-0.425, -0.265, 0),
		E = vec3(-0.425, 0.265, 0),
		F = vec3(-0.265, 0, 0.425),
		G = vec3(0.265, 0, 0.425),
		H = vec3(0.265, 0, -0.425),
		I = vec3(-0.265, 0, -0.425),
		J = vec3(0, -0.425 , -0.265),
		K = vec3(0, -0.425, 0.265),
		L = vec3(0, 0.425, 0.265);

		float v = 0.5;
		A.y += v;
		B.y += v;
		C.y += v;
		D.y += v;
		E.y += v;
		F.y += v;
		G.y += v;
		H.y += v;
		I.y += v;
		J.y += v;
		K.y += v;
		L.y += v;

		A.z += v;
		B.z += v;
		C.z += v;
		D.z += v;
		E.z += v;
		F.z += v;
		G.z += v;
		H.z += v;
		I.z += v;
		J.z += v;
		K.z += v;
		L.z += v;

		A.x -= v/2;
		B.x -= v/2;
		C.x -= v/2;
		D.x -= v/2;
		E.x -= v/2;
		F.x -= v/2;
		G.x -= v/2;
		H.x -= v/2;
		I.x -= v/2;
		J.x -= v/2;
		K.x -= v/2;
		L.x -= v/2;

		sides = std::vector<Square>();
		sides.push_back(Square(G,B,C,H, material));
		sides.push_back(Square(B, L, G, F, material));
		sides.push_back(Square(L,A,B,H, material));
		sides.push_back(Square(C, G, K, F, material));
		sides.push_back(Square(K,C,J,H, material));
		sides.push_back(Square(A,I,H,J, material));
		sides.push_back(Square(A,I,F,D, material));
		sides.push_back(Square(I,J,D,K, material));
		sides.push_back(Square(A,L,E,F, material));
		sides.push_back(Square(E,F,D,K, material));
		
		objects.push_back(new Icosahedron(sides));

		
		//Octahedron - squares
		A = vec3(0.5, 0.5, 0),
		B = vec3(0, 0, 0),	
		C = vec3(-0.5, 0.5, 0),
		D = vec3(0, 1, 0),	
		E = vec3(0, 0.5, 0.5),
		F = vec3(0, 0.5, -0.5);


		//TEST: eltolás
		float val = 0.5f;
		A.x += val;
		B.x += val;
		C.x += val;
		D.x += val;
		E.x += val;
		F.x += val;

		val = -0.2f;
		A.z += val;
		B.z += val;
		C.z += val;
		D.z += val;
		E.z += val;
		F.z += val;

		sides = std::vector<Square>();
		sides.push_back(Square(A,B,E,C, material));
		sides.push_back(Square(C,D,E,A, material));
		sides.push_back(Square(A,B,F,C, material));
		sides.push_back(Square(C,D,F,A, material));
		if(octa) objects.push_back(new Octahedron(sides));


		//Y és Z felcserélve
		A = vec3(-1.0, 0.0, -1.0),
		B = vec3(-1.0, 2.0, -1.0),
		C = vec3(-1.0, 0.0, 1.0),
		D = vec3(-1.0, 2.0, 1.0),
		E = vec3(1.0, 0.0, -1.0),
		F = vec3(1.0, 2.0, -1.0),
		G = vec3(1.0, 0.0, 1.0),
		H = vec3(1.0, 2.0, 1.0);

		
		sides.push_back(Square(E, G, A, C, material)); //padló
		sides.push_back(Square(C, D, A, B, material)); //bal hátsó oldal
		sides.push_back(Square(E, F, A, B, material)); //jobb hátsó
		sides.push_back(Square(F, H, B, D, material)); //tető
		sides.push_back(Square(G,H,E,F, material)); //jobb első oldal
		sides.push_back(Square(G,H,C,D, material)); //bal első oldal
		if(cube) objects.push_back(new Cube(sides));
	}

	void render(std::vector<vec4>& image) { //virt. vil�g lef�nyk�pez�se
		for (int Y = 0; Y < windowHeight; Y++) {
#pragma omp parallel for
			for (int X = 0; X < windowWidth; X++) {
				vec3 color = trace(camera.getRay(X, Y)); //minden pixelnek meghat�rozzuk a sz�n�t
				image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
			}
		}
	}

	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		for (Intersectable* obj : objects) {
			Hit hit = obj->intersect(ray);
			if (hit.t > 0 && (bestHit.t < 0 || bestHit.t > hit.t)) {
				bestHit = hit;
			}
		}
		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}

	//TODO: yet wrong
	Hit secondIntersect(Ray ray) {
		Hit bestHit, secondHit;
		
		for (Intersectable* obj : objects) {
			Hit hit = obj->intersect(ray);
			if (hit.t > 0 && (bestHit.t < 0 || bestHit.t > hit.t))  bestHit = hit;
		}
		
		for (Intersectable* obj : objects) {
			Hit hit = obj->intersect(ray);
			if (hit.t > 0 && (secondHit.t < 0 || secondHit.t == bestHit.t)) {
				secondHit = hit;
			}
		}
		//if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
		if (dot(ray.dir, secondHit.normal) > 0) secondHit.normal = secondHit.normal * (-1);
		return secondHit;

	}



	bool shadowIntersect(Ray ray) {	// for directional lights
		for (Intersectable* object : objects) if (object->intersect(ray).t > 0) return true;
		return false;
	}

	vec3 trace(Ray ray, int depth = 0) {
		//Hit hit = secondIntersect(ray);
		Hit hit = firstIntersect(ray);
		float val = 0.2 * (1 + dot(normalize(hit.normal), normalize(ray.dir)));
		La = vec3(val,val,val);
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