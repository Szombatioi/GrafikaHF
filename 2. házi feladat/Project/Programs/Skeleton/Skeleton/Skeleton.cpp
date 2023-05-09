#include "framework.h"

//TODO:
//nyilatkozat
//kommentek törlése
//forrásmegjelölések

//Szirmay-Kalos László munkássága alapján
struct Material {
	vec3 ka, kd, ks;
	float shininess;
	Material(vec3 _kd, vec3 _ks, float _shininess) : ka(_kd* M_PI), kd(_kd), ks(_ks) { shininess = _shininess; }
};

//Szirmay-Kalos László munkássága alapján
struct Hit {
	float t;
	vec3 position, normal;
	Material* material;
	Hit() { t = -1; }
};

//Szirmay-Kalos László munkássága alapján
struct Ray {
	vec3 start, dir;
	Ray(vec3 _start, vec3 _dir) : start(_start), dir(_dir) {}
};

//Szirmay-Kalos László munkássága alapján
class Intersectable {
protected:
	Material* material;
public:
	virtual Hit intersect(const Ray& ray) = 0;
};

//TODO: alakzatok

//Szirmay-Kalos László munkássága alapján
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
	vec3 direction, Le; //irány, intenzitás
	Light(vec3 _direction, vec3 _Le) {
		direction = normalize(_direction);
		Le = _Le;
	}
};

struct Ray {
	vec3 start, dir;
	//bool out;
};

struct Hit {
	float t;
	vec3 position, normal;
	Material* material;
	Hit() { t = -1; }
};

class Scene {
	std::vector<Intersectable*> objects;
	std::vector<Light*> lights;
	Camera camera;
	vec3 La; //ambiens fény
public:
	void build() {
		vec3 eye = vec3(0, 0, 2), vup = vec3(0, 1, 0), lookat = vec3(0, 0, 0);
		float fov = 45 * M_PI / 180;
		camera.set(eye, lookat, vup, fov);

		La = vec3(0.4f, 0.4f, 0.4f);
		vec3 lightDirection(1, 1, 1), Le(2, 2, 2);
		lights.push_back(new Light(lightDirection, Le));

		//2 material készítése
		vec3 kd1(0.3f, 0.2f, 0.1f), kd2(0.1f, 0.2f, 0.3f), ks(2, 2, 2); //diffúz és spekuláris visszaverõdési tényezõk (kell?)
		Material* material1 = new Material(kd1, ks, 50); //50?
		Material* material2 = new Material(kd2, ks, 50); //50?

		//objects.push_back...
	}

	void render(std::vector<vec4>& image) { //virt. világ lefényképezése
		for (int Y = 0; Y < windowHeight; Y++) {
#pragma omp parallel for
			for (int X = 0; X < windowWidth; X++) {
				vec3 color = trace(camera.getRay(X, Y)); //minden pixelnek meghatározzuk a színét

			}
		}
	}

	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		for (Intersectable* obj : objects) {
			Hit hit = obj->intersect(ray);
			if (hit.t > 0 && (bestHit.t<0 || bestHit.t > hit.t)) bestHit = hit;
		}
		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal *= -1;
		return bestHit;
	}
};


class Cube : public Intersectable {
	std::vector<vec3> points;
	Hit intersect(const Ray& ray) {
		
	}
};
