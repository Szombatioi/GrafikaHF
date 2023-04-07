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

float testValueForMove = 0.05;
float testValueForRotation = 0.05;

vec2 hyProject(vec3 point) {
	return vec2(point.x / (point.z + 1), point.y / (point.z + 1));
}

float hyDot(vec3 p, vec3 q) {
	return p.x * q.x + p.y * q.y - p.z * q.z;
}
vec3 hyNormalize(vec3 vector) {
	return vector / sqrtf(fabs(hyDot(vector, vector)));
}

vec3 hyCross(vec3 v, vec3 w) {
	return vec3(cross(vec3(v.x, v.y, -v.z), vec3(w.x, w.y, -w.z)));
}

// 1. Egy irányra merőleges irány állítása.
vec3 hyPerp(vec3 vector) {
	return hyCross(vector, vec3(0,0,-1));
}

// 2. Adott pontból és sebesség vektorral induló pont helyének és sebesség vektorának számítása t idővel később.
vec3 hyMoveP(vec3 point, vec3 vector, float t) {
	return point * coshf(t) + hyNormalize(vector) * sinhf(t);
}
vec3 hyMoveV(vec3 point, vec3 vector, float t) {
	return point * sinhf(t) + hyNormalize(vector) * sinhf(t);
}

// 3. Egy ponthoz képest egy másik pont irányának és távolságának meghatározása.

float hyDist(vec3 p, vec3 q) {
	return acoshf(-hyDot(p, q));
}

vec3 hyDir(vec3 p, vec3 q) {
	return vec3((q - p * coshf(hyDist(p,q))) / sinhf(hyDist(p, q)));
}

// 4. Egy ponthoz képest adott irányban és távolságra lévő pont előállítása.
vec3 hyProduceP(vec3 point, vec3 vector, float distance) {
	return hyMoveP(point, vector, distance);
}

// 5. Egy pontban egy vektor elforgatása adott szöggel.
vec3 hyRotate(vec3 vector, float phi) {
	vec3 v = hyNormalize(vector);
	return v * cosf(phi) + hyNormalize(hyPerp(v)) * sinf(phi);
}

// 6. Egy közelítő pont és sebességvektorhoz a geometria szabályait teljesítő, közeli pont és sebesség választása.
vec3 hyNearP(vec3 point) {
	return hyNormalize(point);
}

vec3 hyNearV(vec3 point, vec3 vector) {
	float lambda = hyDot(point, vector);
	return vector + lambda * point;
}

//Forrás: https://stackoverflow.com/questions/17134839/how-does-the-map-function-in-processing-work
float map(float value, float istart, float istop, float ostart, float ostop) {
	return ostart + (ostop - ostart) * ((value - istart) / (istop - istart));
}

//Forrás: Szirmay-Kalos László "Háromszög szerkesztő a Beltrami-Poincaré diszk modellben" kódja alapján
//http://cg.iit.bme.hu/portal/sites/default/files/oktatott%20t%C3%A1rgyak/sz%C3%A1m%C3%ADt%C3%B3g%C3%A9pes%20grafika/grafikus%20alap%20hw/sw/poincare.cpp
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
float delta_t;
std::vector<vec2> circlePoints;

struct Circle {
	vec3 center;
	float radius;
	Circle(vec3 c = vec3(0, 0, 1), float r = 0.35f) : center(c), radius(r) {}
	void draw(vec3 color) {
		vec3 direction;
		if (hyDist(center, vec3(0, 0, 1)) <= 0.0001) {
			direction = vec3(1, 0, 0);
		}
		else {
			direction = hyDir(center, vec3(0, 0, 1));
		}
		vec3 perp = hyNormalize(hyCross(center, direction));
		std::vector<vec3> circlePoints;
		
		for (int i = 0; i < ncircleVertices; i++) {
			float angle = 2 * M_PI * i / ncircleVertices;
			vec3 dir = hyNormalize(direction * cosf(angle) + perp * sinf(angle));
			vec3 point = center * coshf(radius) + dir * sinhf(radius);
			circlePoints.push_back(hyNormalize(point));
		}

		renderer->DrawGPU(GL_TRIANGLE_FAN, circlePoints, color);
	}
};

struct Hami {
	vec3 center, direction;
	Circle body, mouth, lEye, rEye, lPup, rPup;
	std::vector<vec2> path;
	Hami(vec3 c = vec3(0,0,1), vec3 d = vec3(1,2,0)) : center(c), direction(d){
		body = Circle(center);
		mouth = Circle(hyNearP(hyProduceP(center, direction, body.radius)), body.radius / 3);
		lEye = Circle(vec3(0,0,0), body.radius / 3);
		rEye = Circle(vec3(0,0,0), body.radius / 3);
		lPup = Circle(vec3(0,0,0), lEye.radius / 2);
		rPup = Circle(vec3(0,0,0), rEye.radius / 2);
	}
	void draw(vec3 color, Hami other) {
		float dist = hyDist(vec3(0, 0, 1), center) + 1;

		body.center = center;
		body.draw(color);
		
		lEye.center = hyNearP(hyProduceP(center, hyNormalize(hyNearV(center, hyRotate(direction, (M_PI / 6) / dist))), body.radius)); //Kell nearPoint? kell nearV?
		rEye.center = hyNearP(hyProduceP(center, hyNormalize(hyNearV(center, hyRotate(direction, -(M_PI / 6) / dist))), body.radius));
		lEye.draw(vec3(1,1,1));
		rEye.draw(vec3(1,1,1));

		lPup.center = hyProduceP(lEye.center, hyNearV(lEye.center, hyDir(body.center, other.body.center)), 3 * lEye.radius / 4);
		rPup.center = hyProduceP(rEye.center, hyNearV(rEye.center, hyDir(body.center, other.body.center)), 3 * rEye.radius / 4);
		lPup.draw(vec3(0,0,1));
		rPup.draw(vec3(0,0,1));

		mouth.center = hyNearP(hyProduceP(center, direction, body.radius));
		mouth.radius = map(sin(delta_t * 10.0f), -1, 1, body.radius / 4, body.radius / 3);
		mouth.draw(vec3(0,0,0));		
	}

	void rotate(bool left) {
		direction = hyNearV(center, hyRotate(direction, testValueForRotation * (left ? -1.0f : 1.0f)));
	}

	void move(float delta) {
		center = hyNearP(hyMoveP(center, direction, delta));
		direction = hyNormalize(hyNearV(center, hyMoveV(center, direction, delta)));
		path.push_back(hyProject(center));
	}
};

Hami piros, zold;

void onInitialization() {
	glLineWidth(2.0f);
	t = delta_t = 0;
	renderer = new Renderer();
	piros = Hami(vec3(0,0,1));
	zold = Hami(vec3(1, 0.75, sqrt(2.5625)), vec3(2,-2,0));

	for (int i = 0; i < ncircleVertices; i++) {
		float angle = M_PI * 2.0f * i / ncircleVertices;
		circlePoints.push_back(vec2(cosf(angle), sinf(angle)));
	}
}

bool keys[256];
void onDisplay() {
	glClearColor(0.5f, 0.5f, 0.5f, 0);
	glClear(GL_COLOR_BUFFER_BIT);
	renderer->DrawGPU(GL_TRIANGLE_FAN, circlePoints, vec3(0, 0, 0));
	renderer->DrawGPU(GL_LINE_STRIP, piros.path, vec3(1, 1, 1));
	renderer->DrawGPU(GL_LINE_STRIP, zold.path, vec3(1, 1, 1));
	
	zold.draw(vec3(0, 1, 0), piros);
	piros.draw(vec3(1,0,0), zold);

	delta_t = glutGet(GLUT_ELAPSED_TIME);
	glutSwapBuffers();
}

void onIdle() {
	t = glutGet(GLUT_ELAPSED_TIME);
	delta_t = t / 1000.0f;
	if (t - 1000/60 > delta_t) { //TODO ????
		int s = (t - delta_t) * 60 / 1000;
		for (int i = 0; i < s; i++) {
			zold.move(testValueForMove);
			zold.rotate(false);
			if (keys['e']) { piros.move(testValueForMove); }
			if (keys['s']) { piros.rotate(true); }
			if (keys['f']) { piros.rotate(false); }
		}
		glutPostRedisplay();
	}
}


void onKeyboard(unsigned char key, int pX, int pY) { keys[key] = true; }
void onKeyboardUp(unsigned char key, int pX, int pY) { keys[key] = false; }
void onMouseMotion(int pX, int pY) {}
void onMouse(int button, int state, int pX, int pY) {}
