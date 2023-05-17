//=============================================================================================
// Masodik hazi feladat: Lehallgatastervezo
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Szombati Oliver Istvan
// Neptun : P37PLU
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

const float epsilon = 0.0001f, delta = 0.01;

float dist_3D(vec3 p1, vec3 p2) {
	vec3 v = p2 - p1;
	return sqrt(dot(v,v));
}

//Forras: Szirmay-Kalos Laszlo kodjai alapjan
struct Material {
	vec3 ka, kd, ks;
	float shininess;
	vec3 diffuseAlbedo;
	Material(vec3 _kd, vec3 _ks, float _shininess, vec3 da) : diffuseAlbedo(da), ka(_kd* M_PI), kd(_kd), ks(_ks) { shininess = _shininess; }
};

//Forras: Szirmay-Kalos Laszlo kodjai alapjan
struct Hit {
	float t;
	vec3 position, normal;
	Material* material;
	Hit() { t = -1; material = nullptr; }
};

//Forras: Szirmay-Kalos Laszlo kodjai alapjan
struct Ray {
	vec3 start, dir;
	Ray(vec3 _start, vec3 _dir) : start(_start), dir(_dir) {}
};

//Forras: Szirmay-Kalos Laszlo kodjai alapjan
class Intersectable {
protected:
	Material* material;
public:
	virtual Hit intersect(const Ray& ray) = 0;
};

//Forras: Szirmay-Kalos Laszlo kodjai alapjan
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

//Forras: Szirmay-Kalos Laszlo kodjai alapjan
struct DLight {
	vec3 direction, Le; //ir�ny, intenzit�s
	DLight(vec3 _direction, vec3 _Le) {
		direction = normalize(_direction);
		Le = _Le;
	}
};

//Forras: Szirmay-Kalos Laszlo kodjai alapjan
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
		return (power / (distance2 * 8 * M_PI));
	}
};

class Triangle : public Intersectable{
	vec3 A, B, C;
	vec3 normal;
	Material* material;
public:
	Triangle(std::vector<vec3> _points, Material* _material){
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

	//Forras: Az eloadasfoliak alapjan
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

		vec3 start_to_point = ray.start - p;
		float cos_square = pow(cosf(angle), 2);
		float a = pow(dot(ray.dir, n), 2) - dot(ray.dir, ray.dir) * cos_square;
		float b = 2.0f * (dot(ray.dir, n) * dot(start_to_point, n) - cos_square * dot(ray.dir, start_to_point));
		float c = dot(n, start_to_point)* dot(n, start_to_point) - dot(start_to_point, start_to_point) * cos_square;
		
		float discr = b * b - 4.0f * a * c;
		if (discr < 0) return hit;
		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;
		float t2 = (-b - sqrt_discr) / 2.0f / a;
		if (t1 <= 0 && t2 <= 0) return hit;

		float d1 = dot(ray.start + ray.dir * t1 - p, n);
		float d2 = dot(ray.start + ray.dir * t2 - p, n);
		if (t1 > 0.0f && 0.0f <= d1 && d1 <= height) {
			hit.t = t1;
		}
		else if (t2 > 0.0f && 0.0f <= d2 && d2 <= height) {
			hit.t = t2;
		}

		vec3 intP = ray.start + hit.t * ray.dir - p;
		float h = dot(intP, n);
		if (h < 0.0 || h > height) return Hit();

		vec3 point = ray.start + ray.dir * hit.t;
		hit.position = point;
		hit.normal = normalize(2*dot((point - p),n)*n - 2*(point - p)*cos_square);
		hit.material = material;
		return hit;
	}
};

struct Bug {
	Cone* cone;
	PLight* light;
	Bug(Cone* c, PLight* l) : cone(c), light(l) {}
	void Move(vec3 to, vec3 normal) {
		cone->p = to;
		cone->n = normalize(normal);
		light->location = to + cone->n * delta;
	}
};

class Square : public Intersectable {
	std::vector<Triangle> triangles; //2
public:
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
	
	Cube(std::vector<vec3> points, std::vector<int> panes, Material *material) {
		sides = std::vector<Square>();
		for (int i = 0; i < panes.size(); i += 4) {
			sides.push_back(Square(points[panes[i]], points[panes[i + 1]], points[panes[i + 2]], points[panes[i+3]], material));
		}
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
	}
};

class Icosahedron : public Intersectable {
	std::vector<Square> sides;
public:
	Icosahedron(std::vector<vec3> points, std::vector<int> panes, Material* material) {
		sides = std::vector<Square>();
		for (int i = 0; i < panes.size(); i += 4) {
			sides.push_back(Square(points[panes[i]], points[panes[i + 1]], points[panes[i + 2]], points[panes[i+3]], material));
		}
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
	Octahedron(std::vector<vec3> points, std::vector<int> panes, Material* material) {
		sides = std::vector<Square>();
		for (int i = 0; i < panes.size(); i += 4) {
			sides.push_back(Square(points[panes[i]], points[panes[i + 1]], points[panes[i + 2]], points[panes[i + 3]], material));
		}
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

//Forras: Szirmay-Kalos Laszlo kodjai alapjan
class Scene {
	std::vector<Intersectable*> objects;
	std::vector<DLight*> lights;
	std::vector<Bug*> bugs;
	Camera camera;
	vec3 La;
public:
	void build() {
		vec3 eye = vec3(2.5f, 1.5f, 2.5f), vup = vec3(0, 1, 0), lookat = vec3(0,0.8,0);
		float fov = 45 * M_PI / 180;
		camera.set(eye, lookat, vup, fov);

		La = vec3(0.0f, 0.0f, 0.0f);
		vec3 lightDirection(0.1,0.1,0.1), Le(1.5f, 1.5f, 1.5f);
		lights.push_back(new DLight(lightDirection, Le));

		vec3 kd1(0.3f, 0.3f, 0.3f), ks(1,1,1);
		Material* material = new Material(kd1, ks, 75, vec3(1,1,1));

		vec3 A, B, C, D, E, F, G, H, I, J, K, L;

		A = vec3(-0.25f, 0.925f, 0.235f),
		B = vec3(0.175f, 0.765f, 0.5f),
		C = vec3(0.175f, 0.235f, 0.5f),
		D = vec3(-0.675f, 0.235f, 0.5f),
		E = vec3(-0.675f, 0.765f, 0.5f),
		F = vec3(-0.515f, 0.5f, 0.925f),
		G = vec3(0.015f, 0.5f, 0.925f),
		H = vec3(0.015f, 0.5f, 0.075f),
		I = vec3(-0.515f, 0.5f, 0.075f),
		J = vec3(-0.25f, 0.075f , 0.235f),
		K = vec3(-0.25f, 0.075f, 0.765f),
		L = vec3(-0.25f, 0.925f, 0.765f);		
		std::vector<vec3> icosaPoints = { A,B,C,D,E,F,G,H,I,J,K,L };
		std::vector<int> icosaPointsIdx = {
			6,1,2,7,
			1,11,6,5,
			11,0,1,7,
			2,6,10,5,
			10,2,9,7,
			0,8,7,9,
			0,8,5,3,
			8,9,3,10,
			0,11,4,5,
			4,5,3,10
		};
		objects.push_back(new Icosahedron(icosaPoints, icosaPointsIdx, material));
		

		A = vec3(1.0f, 0.5f, -0.2f),
		B = vec3(0.5f, 0.0f, -0.2f),
		C = vec3(0.0f, 0.5f, -0.2f),
		D = vec3(0.5f, 1.0f, -0.2f),
		E = vec3(0.5f, 0.5f, 0.3f),	
		F = vec3(0.5f, 0.5f, -0.8f);
		std::vector<vec3> octaPoints = {A,B,C,D,E,F};
		std::vector<int> octaPointsIdx = {
			0,1,4,2,
			2,3,4,0,
			0,1,5,2,
			2,3,5,0
		};
		objects.push_back(new Octahedron(octaPoints, octaPointsIdx, material));


		A = vec3(-1.0, 0.0, -1.0),
		B = vec3(-1.0, 2.0, -1.0),
		C = vec3(-1.0, 0.0, 1.0),
		D = vec3(-1.0, 2.0, 1.0),
		E = vec3(1.0, 0.0, -1.0),
		F = vec3(1.0, 2.0, -1.0),
		G = vec3(1.0, 0.0, 1.0),
		H = vec3(1.0, 2.0, 1.0);
		std::vector<vec3> cubePoints = { A,B,C,D,E,F,G,H };
		std::vector<int> cubePointIdx = {
			4,6,0,2,
			2,3,0,1,
			4,5,0,1,
			5,7,1,3,
			6,7,4,5,
			6,7,2,3
		};
		objects.push_back(new Cube(cubePoints, cubePointIdx, material));

		Cone* c1 = new Cone(vec3(0, 2, 0), vec3(0, -1, 0), 0.2, M_PI / 8, material); //red
		Cone* c2 = new Cone(vec3(-1, 1.5, 0), vec3(1,0,0), 0.2, M_PI / 8, material); //green
		Cone* c3 = new Cone(vec3(0.064, 0.735, 0.359), vec3(0.577, 0.577, -0.577), 0.2, M_PI / 8, material); //blue	

		PLight* p1 = new PLight(c1->p + c1->n * delta, vec3(150,0,0));
		PLight* p2 = new PLight(c2->p + c2->n * delta, vec3(0,150,0));
		PLight* p3 = new PLight(c3->p + c3->n * delta, vec3(0,0,150));

		objects.push_back(c1);
		objects.push_back(c2);
		objects.push_back(c3);
		bugs.push_back(new Bug(c1, p1));
		bugs.push_back(new Bug(c2, p2));
		bugs.push_back(new Bug(c3, p3));
	}

	//Forras: Szirmay-Kalos Laszlo kodjai alapjan
	void render(std::vector<vec4>& image) {
		for (int Y = 0; Y < windowHeight; Y++) {
#pragma omp parallel for
			for (int X = 0; X < windowWidth; X++) {
				vec3 color = trace(camera.getRay(X, Y)); //minden pixelnek meghat�rozzuk a sz�n�t
				image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
			}
		}
	}

	//Forras: Szirmay-Kalos Laszlo kodjai alapjan
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
	
	//Forras: Szirmay-Kalos Laszlo kodjai alapjan
	bool shadowIntersect(Ray ray) {
		for (Intersectable* object : objects) if (object->intersect(ray).t > 0) return true;
		return false;
	}


	vec3 trace(Ray ray, int depth = 0) {
		Hit hit = firstIntersect(ray);
		float val = 0.2 * (1 + dot(normalize(hit.normal), normalize(ray.dir)));
		La = vec3(val,val,val);
		if (hit.t < 0) return vec3(0.0, 0.0, 0.0);
		vec3 outRadiance = hit.material->ka * La;

		for (DLight* light : lights) {
			float cosTheta = dot(hit.normal, light->direction);
			if (cosTheta > 0) {
				outRadiance = outRadiance + light->Le * hit.material->kd * cosTheta;
				vec3 halfway = normalize(-ray.dir + light->direction);
				float cosDelta = dot(hit.normal, halfway);
				if (cosDelta > 0) outRadiance = outRadiance + light->Le * hit.material->ks * powf(cosDelta, hit.material->shininess);
			}
		}

		vec3 outRad(0, 0, 0);
		vec3 N = hit.normal;
		vec3 outDir;
		for (Bug *bug : bugs) {
			PLight *light = bug->light;
			outDir = light->directionOf(hit.position + hit.normal * epsilon);
			Hit shadowHit = firstIntersect(Ray(hit.position + N * epsilon, outDir));
			if (shadowHit.t < 0.0f || shadowHit.t > light->distanceOf(hit.position + hit.normal * epsilon)) {
				double cosThetaL = dot(N, outDir);
				outRad = outRad + hit.material->diffuseAlbedo / M_PI * cosThetaL * light->radianceAt(hit.position + hit.normal * epsilon);
			}
		}

		return outRad + outRadiance;
	}

	void mouseClick(int pX, int pY) {
		Ray ray = camera.getRay(pX, pY);
		Hit hit = firstIntersect(ray);

		Bug* closestBug;
		float dist = INT_MAX;

		for (Bug* b : bugs) {
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

//Forras: Szirmay-Kalos Laszlo kodjai alapjan
const char *vertexSource = R"(
	#version 330
    precision highp float;

	layout(location = 0) in vec2 cVertexPosition;
	out vec2 texcoord;

	void main() {
		texcoord = (cVertexPosition + vec2(1, 1))/2;
		gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1);
	}
)";

//Forras: Szirmay-Kalos Laszlo kodjai alapjan
const char *fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	in  vec2 texcoord;
	out vec4 fragmentColor;

	void main() {
		fragmentColor = texture(textureUnit, texcoord); 
	}
)";

//Forras: Szirmay-Kalos Laszlo kodjaibol
class FullScreenTexturedQuad {
	unsigned int vao;
	Texture texture;
public:
	FullScreenTexturedQuad(int windowWidth, int windowHeight, std::vector<vec4>& image)
		: texture(windowWidth, windowHeight, image) 
	{
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);

		unsigned int vbo;
		glGenBuffers(1, &vbo);

		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
	}

	void Draw() {
		glBindVertexArray(vao);
		gpuProgram.setUniform(texture, "textureUnit");
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
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