//=============================================================================================
// Mintaprogram: Zold haromszog. Ervenyes 2019. osztol.
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

float delta_t = 0.02;

//1. irányra merőleges állítása
vec3 hyPerp(vec3 vector) {
	//vector.z *= -1;
	return cross(vector, vec3(0,0,-1));
}

//2. Adott pontból és sebesség vektorral induló pont helyének és sebesség vektorának számítása t idővel később.
vec3 hyMoveP(vec3 point, vec3 vector, float t) {
	return point * coshf(t) + normalize(vector) * sinhf(t);
}

vec3 hyMoveV(vec3 point, vec3 vector, float t) {
	return point * sinhf(t) + normalize(vector) * coshf(t);
}

//3. Egy ponthoz képest egy másik pont irányának és távolságának meghatározása.
vec3 hyDir(vec3 p, vec3 q) {
	//TODO: szemek irányának meghatározásához
	return vec3((q - p * coshf(delta_t)) / sinhf(delta_t));
}

float hyDot(vec3 p, vec3 q) {
	return p.x*q.x + p.y*q.y - p.z*q.z;
}

float hyDist(vec3 point, vec3 otherPoint) {
	return acoshf(-1.0f * hyDot(otherPoint, point));
}

//4. Egy ponthoz képest adott irányban és távolságra lévő pont előállítása.
vec3 hyProduceP(vec3 point, vec3 dir, float dist) {
	return point * coshf(dist) + normalize(dir) * sinhf(dist);
	//return normalize(dir) * dist + point; //lehet kell visszavetítés
	//vagy lehet a hyMoveP kell
}

//5. Egy pontban egy vektor elforgatása adott szöggel.
vec3 hyRotate(vec3 point, vec3 vector, float angle) {
	vec3 v = normalize(vector);
	return vec3(v * cosf(angle) + hyPerp(v) * sinf(angle));
}

//6. Egy közelítő pont és sebességvektorhoz a geometria szabályait teljesítő, közeli pont és sebesség választása.
vec3 hyInvPoint(vec3 point) {
	point.z = sqrt(point.x * point.x + point.y * point.y + 1);
	return point;
}

vec3 hyInvVector(vec3 point, vec3 vector) {
	float lambda = hyDot(point, vector);
	return vector + lambda * point;
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

	vec2 projectPoincare(vec3 point) {
		return vec2(point.x / (point.z + 1), point.y / (point.z+1));
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
float t;

struct Circle {
	float radius;
	vec3 center;
	std::vector<vec3> points;
	Circle() {}
	Circle(vec3 c, float r) : center(c), radius(r) {}
	
	void draw(vec3 color) {
		vec3 dir;
		if (hyDist(center, vec3(0, 0, 1)) <= 0.01) { dir = vec3(1, 1, 1); } //ha a centertől 0.001-re van
		else dir = hyDir(center, vec3(0,0,1)); //ha nem centerbe van, akkor vector: pos->0,0,1
		
		std::vector<vec3> points;
		//vec3 start = centerPoint;
		//perpendic normalise(cross())

		//vec3 perp = hyPerp(dir);
		for (int i = 0; i < ncircleVertices; i++) {
			float angle = M_PI * 2.0f * i / ncircleVertices;
			float x = center.x + cosf(angle) * radius, 
				  y = center.y + sinf(angle) * radius;
			points.push_back(vec3(x,y,1));

			//vec3 newPoint = hyMoveP(center, dir, radius);
			////dir = hyMoveV(center, dir, radius);
			//dir = hyRotate(center, dir, angle);
			//points.push_back(newPoint);
			
						/*start = movePoint(centerPoint, angle, i);
			points.push_back(start);
			moveVec(centerPoint, angle, i);*/
		}

		renderer->DrawGPU(GL_TRIANGLE_FAN, points, color);
		//renderer->DrawGPU(GL_POINTS, points, color);
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
	Circle body, mouth; //TODO: fölösleges eltárolni a hami száját, szemét, stb. -> legyen rá generáló függvény
	std::vector<vec2> path;
	Hami() {}
	Hami(vec3 c, vec3 d) : center(c), direction(d) {
		body = Circle(center, 0.3f);
		mouth = Circle(hyProduceP(center, direction, body.radius), 0.1);
	}
	void draw(vec3 color) {

		renderer->DrawGPU(GL_LINE_STRIP, path, vec3(1,1,1));
		body.draw(color);
		
		mouth.radius = map(sin(t*10.0f), -1, 1, 0.08, 0.115);
		mouth.draw(vec3(0, 1, 1));
	}
	void rotate(bool left) {
		direction = hyInvVector(center, hyRotate(center, direction, 0.05 * (left ? 1.0f : -1.0f)));
	}

	void move() {
		center = hyMoveP(center, direction, delta_t);
		direction = hyInvVector(center, hyMoveV(center, direction, 0.1));
		center = hyInvPoint(center);
		path.push_back(renderer->projectPoincare(center));

		//itt írjuk felül a testrészeket
		body.center = center;
		mouth.center = hyProduceP(center, direction, body.radius);
	}
};
Hami pirosHami, zoldHami;
std::vector<vec2> circlePoints;
vec3 testPoint, testDir;

void onInitialization() {
	renderer = new Renderer();
	//pirosHami = Hami(vec3(0,0,1), vec3(1,2,0));
	pirosHami = Hami(vec3(0,0,1), vec3(0,1,0));
	zoldHami = Hami(vec3(2,2,3), vec3(2,-2,0));

	for (int i = 0; i < ncircleVertices; i++) {
		float angle = M_PI * 2.0f * i / ncircleVertices;
		circlePoints.push_back(vec2(cosf(angle), sinf(angle)));
	}

	/*testPoint = vec3(0,0,1);
	testDir = vec3(1,1,0);*/

	/*vec3 start = vec3(2, 2, 3),
		direction = vec3(-2,2,0);
	vec3 point = start;
	for (int i = 0; i < 100; i++) {
		point = hyMoveP(point, direction, 0.1);
		direction = hyMoveV(point, direction, 0.1);
		line.push_back(point);
	}
	glPointSize(10.0f);*/
	//glPointSize(10.0f);
}
bool keys[256];
void onDisplay() {
	glClearColor(0.5f,0.5f,0.5f,0);
	glClear(GL_COLOR_BUFFER_BIT);
	renderer->DrawGPU(GL_TRIANGLE_FAN, circlePoints, vec3(0, 0, 0));

	
	pirosHami.draw(vec3(1, 0, 0));
	zoldHami.draw(vec3(0, 1, 0));

	glutSwapBuffers();
}

void onIdle() {
	t = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;
	
	////TODO ezt törölni
	glClearColor(0.5f,0.5f,0.5f,0);
	glClear(GL_COLOR_BUFFER_BIT);
	renderer->DrawGPU(GL_TRIANGLE_FAN, circlePoints, vec3(0, 0, 0));
	pirosHami.draw(vec3(1, 0, 0));
	zoldHami.draw(vec3(0, 1, 0));
	/*std::vector<vec3> p;
	p.push_back(testPoint);
	renderer->DrawGPU(GL_POINTS, p, vec3(1, 0, 0));*/
	glutSwapBuffers();
	//printf("%.2f, %.2f, %.2f\n", testPoint.x, testPoint.y, testPoint.z);
	if (keys['e']) {
		pirosHami.move();
		/*testPoint = hyMoveP(testPoint, testDir, 0.02);
		testDir = hyInvVector(testPoint, hyMoveV(testPoint, testDir, 0.1));
		testPoint = hyInvPoint(testPoint);

		printf("Valid: %.2f\n\t%.2f\n", hyDot(testPoint, testPoint), hyDot(testPoint, testDir));
		puts("-----");*/
	}

	

	if (keys['s']) {
		pirosHami.rotate(true);
		/*testDir = hyInvVector(testPoint, hyRotate(testPoint, testDir, 0.05));
		printf("Rotate: %.2f,%.2f,%.2f\n", testDir.x, testDir.y, testDir.z);*/
		
	}
	if (keys['f']) {
		pirosHami.rotate(false);
		/*testDir = hyInvVector(testPoint, hyRotate(testPoint, testDir,-0.05));
		printf("Rotate: %.2f,%.2f,%.2f\n", testDir.x, testDir.y, testDir.z);*/
	}
}


void onKeyboard(unsigned char key, int pX, int pY) { keys[key] = true; }
void onKeyboardUp(unsigned char key, int pX, int pY) { keys[key] = false; }
void onMouseMotion(int pX, int pY) {}
void onMouse(int button, int state, int pX, int pY) {}