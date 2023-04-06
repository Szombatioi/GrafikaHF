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

float testValueForMove = 0.02;

////A vektorok normalizálása mindig a hívó feladata! Ne adjunk vissza feleslegesen normalizált vektort

vec2 hyProject(vec3 point) {
	return vec2(point.x / (point.z + 1), point.y / (point.z + 1));
}

float hyDot(vec3 p, vec3 q) {
	return p.x * q.x + p.y * q.y - p.z * q.z;
}
vec3 hyNormalize(vec3 vector) {
	return vector / sqrtf(fabs(hyDot(vector, vector)));
	//return vector / hyDot(vector, vector);
}

vec3 hyCross(vec3 v, vec3 w) {
	return vec3(cross(vec3(v.x, v.y, -v.z), vec3(w.x, w.y, -w.z)));
}

// 1. Egy irányra merőleges irány állítása.
vec3 hyPerp(vec3 vector) {
	return hyCross(vector, vec3(0,0,-1)); //TODO
}

// 2. Adott pontból és sebesség vektorral induló pont helyének és sebesség vektorának számítása t idővel később.
vec3 hyMoveP(vec3 point, vec3 vector, float t) {
	/*vec3 res = point * coshf(t) + hyNormalize(vector) * sinhf(t);
	printf("\t%.2f %.2f %.2f\n", res.x, res.y, res.z);
	return res;*/
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
	return point * coshf(distance) + hyNormalize(vector) * sinhf(distance);
	//vector-t változtatni kellene itt?
}

// 5. Egy pontban egy vektor elforgatása adott szöggel.
vec3 hyRotate(vec3 vector, float phi) {
	vec3 v = hyNormalize(vector);
	return vec3(v * cosf(phi) + hyPerp(v) * sinf(phi));
}

// 6. Egy közelítő pont és sebességvektorhoz a geometria szabályait teljesítő, közeli pont és sebesség választása.
////Ha mozog a pont, akkor mindig vissza kell dobnunk a síkra
vec3 hyNearP(vec3 point) {
	point.z = sqrt(point.x * point.x + point.y * point.y + 1); ///Ehelyett centrális visszavetítés kéne?
	return point;
}

vec3 hyNearV(vec3 point, vec3 vector) {
	float lambda = hyDot(point, vector);
	return vector + lambda * point;
}

float map(float value, float istart, float istop, float ostart, float ostop) {
	return ostart + (ostop - ostart) * ((value - istart) / (istop - istart));
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
float delta_t;
std::vector<vec2> circlePoints;

struct Circle {
	vec3 center;
	float radius;
	Circle(vec3 c = vec3(0, 0, 1), float r = 0.4f) : center(c), radius(r) {}
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
		mouth = Circle(vec3(0,0,0), 0.1f);
		lEye = Circle(vec3(0, 0, 0), 0.08f);
	}
	void draw(vec3 color) {
		renderer->DrawGPU(GL_LINE_STRIP, path, vec3(1, 1, 1));
		body.center = center;
		body.draw(color);

		//mouth
		mouth.center = hyProduceP(center, direction, body.radius);
		mouth.radius = map(sin(delta_t), -1, 1, 0.05, 0.08);
		mouth.draw(vec3(0, 0, 1));

		lEye.center = hyNearP(hyProduceP(center, (hyRotate(direction, 0.5)), body.radius));
		lEye.draw(vec3(1, 1, 1));

	}

	void rotate(bool left) {
		direction = hyNearV(center, hyRotate(direction, 0.05 * (left ? -1.0f : 1.0f)));
	}

	void move(float delta) {
		center = hyMoveP(center, direction, delta);
		direction = hyNormalize(hyNearV(center, hyMoveV(center, direction, delta)));
		center = hyNearP(center);
		path.push_back(hyProject(center));

		//printf("%.2f %.2f %.2f\n", center.x, center.y, center.z);
	}
};

Hami piros, zold;

void onInitialization() {
	glLineWidth(2.0f);
	t = delta_t = 0;
	renderer = new Renderer();

	piros = Hami(vec3(0,0,1));
	zold = Hami(vec3(2, 2, 3), vec3(2,-2,0));

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
	
	piros.draw(vec3(1,0,0));
	zold.draw(vec3(0, 1, 0));

	glutSwapBuffers();
}

void onIdle() {
	t = glutGet(GLUT_ELAPSED_TIME);
	//printf("%.3f\n", t - delta_t);
	if (t - delta_t > 60) {
		delta_t = t;
	}
	/*if (t - delta_t > 15) {
		if (keys['e']) { piros.move(0.01); }
		glutPostRedisplay();
		delta_t = t;
		
	}*/
	glutPostRedisplay();

	zold.move(testValueForMove);
	zold.rotate(false);

	if (keys['e']) { piros.move(testValueForMove); }
	if (keys['s']) { piros.rotate(true); }
	if (keys['f']) { piros.rotate(false); }
	

}


void onKeyboard(unsigned char key, int pX, int pY) { keys[key] = true; }
void onKeyboardUp(unsigned char key, int pX, int pY) { keys[key] = false; }
void onMouseMotion(int pX, int pY) {}
void onMouse(int button, int state, int pX, int pY) {}
