//=============================================================================================
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Szombati Olivér
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


//TODO: 
//kommentek törlése
//források megjelölése
//vektor return -> normalize + approx
//pont return --> approx
//szemek összecsúsznak
//kör kirajzolása

float delta_t = 0.05; //0.02 volt
int last_time;


float hyDot(vec3 p, vec3 q) {
	return p.x*q.x + p.y*q.y - p.z*q.z;
}

vec3 hyNormalize(vec3 point, vec3 vector){
	//return vector / acoshf(-1.0f * hyDot(point, vector)); //vec3(0,0,0) kellhet 1. dot vektornak
	return normalize(vector);
}

vec3 hyCross(vec3 v, vec3 w){
	vec3 res = cross(v,w);
	return res;
}

float hyDot(vec3 p, vec3 q) {
	return p.x*q.x + p.y*q.y - p.z*q.z;
}

vec3 hnorm(vec3 vec) { return vec / sqrtf(fabs(hyDot(vec, vec))); }


vec2 hyProject(vec3 point) {
	return vec2(point.x / (point.z + 1), point.y / (point.z+1));
}

vec3 hyNormalize(vec3 vector){
	//return vector / 
	return normalize(vector); //todo
}

//1. irányra merőleges állítása
vec3 hyPerp(vec3 vector) {
	//vector.z *= -1;
	return normalize(cross(vector, vec3(0,0,-1)));
}

//2. Adott pontból és sebesség vektorral induló pont helyének és sebesség vektorának számítása t idővel később.
vec3 hyMoveP(vec3 point, vec3 vector, float t) {
<<<<<<< HEAD
//	return point * coshf(t) + normalize(vector) * sinhf(t);
	return point * coshf(t) + hyNormalize(point, vector) * sinhf(t);
=======
	return point * coshf(t) + hyNormalize(vector) * sinhf(t);
>>>>>>> newB
}

vec3 hyMoveV(vec3 point, vec3 vector, float t) {
	return hyNormalize(point, point * sinhf(t) + hyNormalize(point, vector) * coshf(t));
}

//3. Egy ponthoz képest egy másik pont irányának és távolságának meghatározása.
vec3 hyDir(vec3 p, vec3 q) {
	//TODO: szemek irányának meghatározásához
	return vec3((q - p * coshf(0.0001)) / sinhf(0.0001)); //0.0001 -> precizitás? t -> 0 ? 
}

<<<<<<< HEAD

=======
>>>>>>> newB
float hyDist(vec3 point, vec3 otherPoint) {
	return acoshf(-1.0f * hyDot(otherPoint, point));
}

//4. Egy ponthoz képest adott irányban és távolságra lévő pont előállítása.
vec3 hyProduceP(vec3 point, vec3 dir, float dist) {
	return point * coshf(dist) + hyNormalize(point, dir) * sinhf(dist);
	//return normalize(dir) * dist + point; //lehet kell visszavetítés
	//vagy lehet a hyMoveP kell
}

//5. Egy pontban egy vektor elforgatása adott szöggel.
vec3 hyRotate(vec3 point, vec3 vector, float angle) {
	vec3 v = hyNormalize(point, vector);
	return hyNormalize(point, vec3(v * cosf(angle) + hyPerp(v) * sinf(angle)));
}

//6. Egy közelítő pont és sebességvektorhoz a geometria szabályait teljesítő, közeli pont és sebesség választása.
vec3 hyInvPoint(vec3 point) {
	point.z = sqrt(point.x * point.x + point.y * point.y + 1);
	return point;
}

vec3 hyInvVector(vec3 point, vec3 vector) {
	float lambda = hyDot(point, vector);
	return hyNormalize(point, vector + lambda * point);
}

class Renderer : public GPUProgram {

	const char* const vertexSource = R"(
		#version 330
		precision highp float;
		layout(location = 0) in vec2 vertexPosition;	// Attrib Array 0

		void main() { gl_Position = vec4(vertexPosition.x, vertexPosition.y, 0, 1); }	
	)";

	const char* const fragmentSource = R"(
		#version 330
		precision highp float;
		uniform vec3 color;
		out vec4 fragmentColor;	

		void main() { fragmentColor = vec4(color, 1); }
	)";

	unsigned int vao, vbo;
public:
	Renderer() {
		glViewport(0, 0, windowWidth, windowHeight);

		create(vertexSource, fragmentSource, "outColor");
		glGenVertexArrays(1, &vao); glBindVertexArray(vao);
		glGenBuffers(1, &vbo);	glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
	}

	std::vector<vec2> projectPoincare(std::vector<vec3> points) {
		std::vector<vec2> res;
		for (auto p : points) res.push_back(vec2(p.x / (p.z + 1), p.y / (p.z + 1)));
		return res;
	}

	void DrawGPU(int type, std::vector<vec2> vertices, vec3 color) {
		setUniform(color, "color");
		glBindVertexArray(vao);
		glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(vec2), &vertices[0], GL_STATIC_DRAW);
		glDrawArrays(type, 0, vertices.size());
	}

	void DrawGPU(int type, std::vector<vec3> vertices, vec3 color) { DrawGPU(type, projectPoincare(vertices), color); }

	~Renderer() {
		glDeleteBuffers(1, &vbo);
		glDeleteVertexArrays(1, &vao);
	}
};
Renderer* renderer;
const int ncircleVertices = 30;
long t;

struct Circle {
	float radius;
	vec3 center;
	//std::vector<vec3> points;
	//std::vector<vec2> points;
	Circle() {}
	Circle(vec3 c, float r) : center(c), radius(r) {}
	//Circle(vec2 c, float r) : center(vec3(c.x, c.y, 0)), radius(r){}
	
	void draw(vec3 color) {
		std::vector<vec3> points;
<<<<<<< HEAD
		//std::vector<vec2> points;
		//vec3 start = centerPoint;
		//perpendic normalise(cross())
=======
>>>>>>> newB

		vec3 direction;
		if (hyDist(center, vec3(0, 0, 1)) <= 0.0001) {
			direction = vec3(1, 0, 0);
		} else {
			direction = hyDir(center, vec3(0, 0, 1));
		}
		auto perpendicular = hnorm(hyCross(center, direction));
		for (int i = 0; i < ncircleVertices; i++) {
			float angle = M_PI * 2.0f * i / ncircleVertices;
			auto dir = hnorm(direction * cosf(angle) + perpendicular * sinf(angle));
			auto point = center * coshf(radius) + dir * sinhf(radius);
			points.push_back(hnorm(point));
		}

		renderer->DrawGPU(GL_TRIANGLE_FAN, points, color);
	}
};

//source: processing.org
float map(float value,
	float istart,
	float istop,
	float ostart,
	float ostop) {
	return ostart + (ostop - ostart) * ((value - istart) / (istop - istart));
}

struct Hami {
	vec3 center, direction;
	Circle body, mouth, lEye, rEye, lPup, rPup;
	std::vector<vec2> path;
	Hami() {}
	Hami(vec3 c, vec3 d) : center(c), direction(d) {
		body = Circle(center, 0.3f);
		mouth = Circle(hyProduceP(center, direction, body.radius), 0.1);
		lEye = Circle(hyProduceP(center, hyRotate(center, direction, 0.5), body.radius), 0.08);
		rEye = Circle(hyProduceP(center, hyRotate(center, direction, -0.5), body.radius), 0.08);
		lPup = Circle(hyProduceP(lEye.center, direction, 0), 0.04);
		rPup = Circle(hyProduceP(rEye.center, direction, 0), 0.04);
	}
	void draw(vec3 color, Hami other) {

<<<<<<< HEAD
		//renderer->DrawGPU(GL_LINE_STRIP, path, vec3(1,1,1));

		//itt írjuk felül a testrészeket
=======
>>>>>>> newB
		body.center = center;
		mouth.center = hyProduceP(center, direction, body.radius);
		lEye.center = hyProduceP(center, hyRotate(center, direction, 0.5), body.radius);
		rEye.center = hyProduceP(center, hyRotate(center, direction, -0.5), body.radius);
		lPup.center = hyProduceP(lEye.center, hyDir(center, other.lEye.center), body.radius / 8);
		rPup.center = hyProduceP(rEye.center, hyDir(center, other.rEye.center), body.radius / 8);

		body.draw(color);

		lEye.draw(vec3(1,1,1));
		rEye.draw(vec3(1, 1, 1));

		lPup.draw(vec3(0, 0, 1));
		rPup.draw(vec3(0, 0, 1));

		mouth.radius = map(sin(t * 5.0f), -1, 1, 0.08, 0.09);
		mouth.draw(vec3(0,0,0));
	}
	void rotate(bool left) {
		direction = hyInvVector(center, hyRotate(center, direction, 0.05 * (left ? 1.0f : -1.0f)));
	}

	void move() {
		center = hyMoveP(center, direction, delta_t);
		direction = hyInvVector(center, hyMoveV(center, direction, 0.1));
		center = hyInvPoint(center);
<<<<<<< HEAD
		path.push_back(renderer->projectPoincare(center));
=======
		path.push_back(hyProject(center));
>>>>>>> newB
	}
};

Hami pirosHami, zoldHami;
std::vector<vec2> circlePoints;
vec3 testPoint, testDir;

void onInitialization() {
<<<<<<< HEAD
	last_time = 0;
	t = 0;
=======
	glLineWidth(2.0f);
>>>>>>> newB
	renderer = new Renderer();
	//pirosHami = Hami(vec3(0,0,1), vec3(1,2,0));
	pirosHami = Hami(vec3(0,0,1), vec3(0,1,0));
	zoldHami = Hami(vec3(0.8, 0.5, sqrt(0.89)), vec3(2,-2,0));

	for (int i = 0; i < ncircleVertices; i++) {
		float angle = M_PI * 2.0f * i / ncircleVertices;
		circlePoints.push_back(vec2(cosf(angle), sinf(angle)));
	}
}

bool keys[256];
void onDisplay() {
	glClearColor(0.5f,0.5f,0.5f,0);
	glClear(GL_COLOR_BUFFER_BIT);
	renderer->DrawGPU(GL_TRIANGLE_FAN, circlePoints, vec3(0, 0, 0));
	renderer->DrawGPU(GL_LINE_STRIP, pirosHami.path, vec3(1,1,1));
	renderer->DrawGPU(GL_LINE_STRIP, zoldHami.path, vec3(1,1,1));
	pirosHami.draw(vec3(1, 0, 0), zoldHami);
	zoldHami.draw(vec3(0, 1, 0), pirosHami);

	glutSwapBuffers();
}

void onIdle() {
	t = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;
<<<<<<< HEAD

	glutPostRedisplay();
	//glutPostRedisplay();
	// if(true || t - last_time > 15){
	// 	zoldHami.move();
	// 	zoldHami.rotate(false);

	// 	if (keys['e']) { pirosHami.move(); }
	// 	if (keys['s']) { pirosHami.rotate(true); }
	// 	if (keys['f']) { pirosHami.rotate(false); }

	// 	glutPostRedisplay();
	// 	last_time = t;
	// }

	zoldHami.move();
	zoldHami.rotate(false);

	if (keys['e']) { pirosHami.move(); }
	if (keys['s']) { pirosHami.rotate(true); }
	if (keys['f']) { pirosHami.rotate(false); }
=======
	glutPostRedisplay();
	
	zoldHami.move();
	zoldHami.rotate(false);

	if (keys['e']) {pirosHami.move();}
	if (keys['s']) {pirosHami.rotate(true);}
	if (keys['f']) {pirosHami.rotate(false);}
>>>>>>> newB
}


void onKeyboard(unsigned char key, int pX, int pY) { keys[key] = true; }
void onKeyboardUp(unsigned char key, int pX, int pY) { keys[key] = false; }
void onMouseMotion(int pX, int pY) {}
void onMouse(int button, int state, int pX, int pY) {}
