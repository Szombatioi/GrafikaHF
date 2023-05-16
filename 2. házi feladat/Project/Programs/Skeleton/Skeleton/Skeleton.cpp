#include "framework.h"
#include <algorithm>

const float epsilon = 0.0001f;
const int maxdepth = 10;
//TODO:
//nyilatkozat
//kommentek torlese
//forr�smegjel�l�sek
//Cone
//delete unnecessary classes
//secondIntersect

float dist_3D(vec3 p1, vec3 p2) {
	vec3 v = p2 - p1;
	return sqrt(dot(v,v));
}

struct Material {
	vec3 ka, kd, ks;
	float shininess;
	vec3 diffuseAlbedo;
	Material(vec3 _kd, vec3 _ks, float _shininess, vec3 da) : diffuseAlbedo(da), ka(_kd* M_PI), kd(_kd), ks(_ks) { shininess = _shininess; }
};

struct Hit {
	float t;
	vec3 position, normal;
	Material* material;
	Hit() { t = -1; material = nullptr; }
};

struct Ray {
	vec3 start, dir;
	Ray(vec3 _start, vec3 _dir) : start(_start), dir(_dir) {}
};

class Intersectable {
protected:
	Material* material;
public:
	virtual Hit intersect(const Ray& ray) = 0;
};

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
};

struct DLight {
	vec3 direction, Le; //ir�ny, intenzit�s
	DLight(vec3 _direction, vec3 _Le) {
		direction = normalize(_direction);
		Le = _Le;
	}
};

struct PLight {
	vec3 location;
	vec3 power;

	PLight(vec3 _location, vec3 _power) {
		location = _location;
		power = _power;
	}
	double distanceOf(vec3 point) {
		return length(location - point);
	}
	vec3 directionOf(vec3 point) {
		return normalize(location - point);
	}
	vec3 radianceAt(vec3 point) {
		double distance2 = dot(location - point, location - point);
		if (distance2 < epsilon) distance2 = epsilon;
		return power / (distance2 * 8 * M_PI);
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

struct Cone : public Intersectable {
	vec3 p, n;
	float height, angle;
	Material* material;

	Cone(vec3 _p, vec3 _n, float h, float a, Material* mat) : p(_p), n(normalize(_n)), material(mat) { height = h; angle = a;}
	Hit intersect(const Ray& ray) {
		Hit hit;
		float cos_square = pow(cosf(angle), 2);
		//float a = pow(dot(ray.dir, n), 2) - dot(ray.dir, ray.dir) * cos_square; //TODO: lehet (dir*n)^2
		//float b = (dot(ray.dir, ray.start) - dot(ray.dir, p)) * (2 * dot(n, n) - 2 * cos_square);
		//float c = dot(n, n) * (dot(ray.start, ray.start) - dot(p,p) - 2*dot(ray.start, p)) - (dot(ray.start, ray.start) + dot(p,p) - 2*dot(ray.start, p))*cos_square;

		vec3 co = ray.start - p;

		float a = dot(ray.dir, n) * dot(ray.dir, n) - dot(ray.dir, ray.dir) * cos_square;
		float b = 2. * (dot(ray.dir, n) * dot(co, n) - dot(ray.dir, co) * cos_square);
		float c = dot(co, n) * dot(co, n) - dot(co, co) * cos_square;
		
		float discr = b * b - 4.0f * a * c;
		if (discr < 0) return hit;
		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;
		float t2 = (-b - sqrt_discr) / 2.0f / a;
		if (t1 <= 0 && t2 <= 0) return hit;
		hit.t = (t2 >= 0) ? t2 : t1;

		vec3 cp = ray.start + hit.t * ray.dir - p;
		float h = dot(cp, n);
		if (h <= 0.0 || h >= height) return Hit();

		
		vec3 point = ray.start + ray.dir * hit.t;
		hit.position = point;
		//hit.normal = normalize(2*((point - p) * n)*n - 2*(point - p)*cos_square);
		hit.normal = normalize(cp * dot(n, cp) / dot(cp,cp) - n);
		hit.material = material;
		return hit;
	}
};

struct Bug {
	Cone* cone;
	PLight* light;
//public:
	Bug(Cone* c, PLight* l) : cone(c), light(l) {}
	void Move(vec3 to, vec3 normal) {
		cone->p = to;
		cone->n = normalize(normal);

		light->location = to + cone->n * 0.05;
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
		Hit bestHit, secondBestHit;
		for (Square s : sides) {
			Hit hit = s.intersect(ray);
			if (hit.t > 0 && (bestHit.t < 0 || bestHit.t > hit.t)) {
				secondBestHit = bestHit;
				bestHit = hit;
			}
		}
		return secondBestHit;
		//return bestHit;
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

class Scene {
	std::vector<Intersectable*> objects;
	std::vector<DLight*> lights;
	std::vector<PLight*> coneLights;
	//std::vector<Cone*> cones;
	std::vector<Bug*> bugs;
	Camera camera;
	vec3 La;
public:
	void build() {
		//vec3 eye = vec3(-4,1.5,4), vup = vec3(0, 1, 0), lookat = vec3(0,0,0); //tests
		vec3 eye = vec3(2.5f, 1.5f, 2.5f), vup = vec3(0, 1, 0), lookat = vec3(0,0.8,0);
		float fov = 45 * M_PI / 180;
		camera.set(eye, lookat, vup, fov);

		La = vec3(0.0f, 0.0f, 0.0f);
		vec3 lightDirection(0.1,0.1,0.1), Le(1.5f, 1.5f, 1.5f);
		//vec3 lightDirection(2,0.5,3), Le(1.5f, 1.5f, 1.5f);
		lights.push_back(new DLight(lightDirection, Le));

		vec3 kd1(0.3f, 0.3f, 0.3f), ks(1,1,1);
		Material* material = new Material(kd1, ks, 75, vec3(1.0,1.0,1.0)); //50 helyett 75

		

		vec3 A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T;
		std::vector<Square> sides;
		std::vector<Triangle> triangles;

		A = vec3(0.0f, 0.425f, -0.265f),
		B = vec3(0.425f, 0.265f, 0.0f),
		C = vec3(0.425f, -0.265f, 0.0f),
		D = vec3(-0.425f, -0.265f, 0.0f),
		E = vec3(-0.425f, 0.265f, 0.0f),
		F = vec3(-0.265f, 0.0f, 0.425f),
		G = vec3(0.265f, 0.0f, 0.425f),
		H = vec3(0.265f, 0.0f, -0.425f),
		I = vec3(-0.265f, 0.0f, -0.425f),
		J = vec3(0.0f, -0.425f , -0.265f),
		K = vec3(0.0f, -0.425f, 0.265f),
		L = vec3(0.0f, 0.425f, 0.265f);

		float v = 0.5;
		A.y += v;  A.z += v;  A.x -= v / 2;
		B.y += v;  B.z += v;  B.x -= v / 2;
		C.y += v;  C.z += v;  C.x -= v / 2;
		D.y += v;  D.z += v;  D.x -= v / 2;
		E.y += v;  E.z += v;  E.x -= v / 2;
		F.y += v;  F.z += v;  F.x -= v / 2;
		G.y += v;  G.z += v;  G.x -= v / 2;
		H.y += v;  H.z += v;  H.x -= v / 2;
		I.y += v;  I.z += v;  I.x -= v / 2;
		J.y += v;  J.z += v;  J.x -= v / 2;
		K.y += v;  K.z += v;  K.x -= v / 2;
		L.y += v;  L.z += v;  L.x -= v / 2;

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
		objects.push_back(new Octahedron(sides));


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
		objects.push_back(new Cube(sides));


		/*vec3 kd2(1.0f, 0.3f, 0.3f);
		Material* material2 = new Material(kd2, ks, 50);
		objects.push_back(new Sphere(lightDirection, 0.1, material2));*/

		vec3 cone1_pos(0, 2, 0);
		vec3 cone1_dir(0, -1, 0);

		vec3 cone2_pos(-1,1, 0);
		vec3 cone2_dir(1, 0, 0);

		vec3 cone3_pos(0.67, 0.67, -0.03);
		vec3 cone3_dir(0.25, 0.25, 0.25);


		Cone* c1 = new Cone(cone1_pos, cone1_dir, 0.2, M_PI / 8, material); //red
		Cone* c2 = new Cone(cone2_pos, cone2_dir, 0.2, M_PI / 8, material); //green
		Cone* c3 = new Cone(cone3_pos, cone3_dir, 0.2, M_PI / 8, material); //blue

		vec3 cone1Light_pos(cone1_pos + cone1_dir * 0.05);
		vec3 cone1Light_pow(50,0,0);
		vec3 cone2Light_pos(cone2_pos + cone2_dir * 0.05);
		vec3 cone2Light_pow(0,50,0);
		vec3 cone3Light_pos(cone3_pos + cone3_dir * 0.05);
		vec3 cone3Light_pow(0,0,50);
		
		PLight* p1 = new PLight(cone1Light_pos, cone1Light_pow);
		PLight* p2 = new PLight(cone2Light_pos, cone2Light_pow);
		PLight* p3 = new PLight(cone3Light_pos, cone3Light_pow);

		coneLights.push_back(p1);
		coneLights.push_back(p2);
		coneLights.push_back(p3);

		objects.push_back(c1);
		//cones.push_back(c1);
		bugs.push_back(new Bug(c1, p1));
		
		objects.push_back(c2);
		//cones.push_back(c2);
		bugs.push_back(new Bug(c2, p2));

		objects.push_back(c3);
		//cones.push_back(c3);
		bugs.push_back(new Bug(c3, p3));


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

	bool shadowIntersect(Ray ray) {	// for directional lights
		for (Intersectable* object : objects) if (object->intersect(ray).t > 0) return true;
		return false;
	}

	Hit clickTrace(Ray ray) {
		Hit hit = firstIntersect(ray);
		return hit;
	}


	//TODO: lehet tényleg másik raytrace fv kell a point lights-nak!
	vec3 trace(Ray ray, int depth = 0) {
		Hit hit = firstIntersect(ray);
		float val = 0.2 * (1 + dot(normalize(hit.normal), normalize(ray.dir)));
		La = vec3(val,val,val);
		if (hit.t < 0) return vec3(0.0, 0.0, 0.0);
		vec3 outRadiance = hit.material->ka * La;

		for (DLight* light : lights) {
			//Ray shadowRay(hit.position + hit.normal * epsilon, light->direction);
			float cosTheta = dot(hit.normal, light->direction);
			if (cosTheta > 0 /* && !shadowIntersect(shadowRay) */ ) {	// shadow computation
				outRadiance = outRadiance + light->Le * hit.material->kd * cosTheta;
				vec3 halfway = normalize(-ray.dir + light->direction);
				float cosDelta = dot(hit.normal, halfway);
				if (cosDelta > 0) outRadiance = outRadiance + light->Le * hit.material->ks * powf(cosDelta, hit.material->shininess);
			}



		}

		vec3 outRad(0, 0, 0);
		if (hit.t < 0 || depth >= maxdepth) return outRad;
		vec3 N = hit.normal;
		vec3 outDir;
		for (Bug *bug : bugs) {
			PLight *light = bug->light;
			outDir = light->directionOf(hit.position);
			Hit shadowHit = firstIntersect(Ray(hit.position + N * epsilon, outDir));
			if (shadowHit.t < epsilon || shadowHit.t > light->distanceOf(hit.position)) {	// if not in shadow
				double cosThetaL = dot(N, outDir);
				if (cosThetaL >= epsilon) {
					outRad = outRad + hit.material->diffuseAlbedo / M_PI * cosThetaL * light->radianceAt(hit.position);
				}
			}
		}

		return outRadiance + outRad;
	}

	void mouseClick(int pX, int pY) {
		//printf("%d, %d\n", pX, pY);


		Ray ray = camera.getRay(pX, pY);
		Hit hit = clickTrace(ray);

		Bug* closestBug;
		float dist = INT_MAX;

		for (Bug* b : bugs) { //ezzel kiválasztjuk a legközelebbi Cone-t
			float d = dist_3D(b->cone->p, hit.position);
			if (d < dist) {
				dist = d;
				closestBug = b;
			}
		}

		closestBug->Move(hit.position, normalize(hit.normal));
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
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
		scene.mouseClick(pX, windowHeight - pY);
		std::vector<vec4> image(windowWidth * windowHeight);
		scene.render(image);

		delete fullScreenTexturedQuad;
		fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);
	}

	glutPostRedisplay();
}
void onKeyboard(unsigned char key, int pX, int pY){}
void onKeyboardUp(unsigned char key, int pX, int pY){}
void onMouseMotion(int pX, int pY){}
void onIdle(){}