//=============================================================================================
// Mintaprogram: Z�ld h�romsz�g. Ervenyes 2019. osztol.
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

//TODO: kommentek törlése, nyilatkozat elolvasása, források megjelölése
//TODO: vetített koordináták fv., azokat kell a vectorba tenni

class Geometria {
	/*
	1. Egy irányra merőleges irány állítása.
	2. Adott pontból és sebesség vektorral induló pont helyének és sebesség vektorának számítása t idővel később.
	3. Egy ponthoz képest egy másik pont irányának és távolságának meghatározása.
	4. Egy ponthoz képest adott irányban és távolságra lévő pont előállítása.
	5. Egy pontban egy vektor elforgatása adott szöggel.
	6. Egy közelítő pont és sebességvektorhoz a geometria szabályait teljesítő, közeli pont és sebesség választása.
	*/

	//1.
	static vec3 iranyraMeroleges(vec3 vector) {
		vec3 temp(0, 0, -1);
		return cross(vector, temp);
	}

	//2.
	static vec3 pontHelye(vec3 point, vec3 vector, float delta_t) {
		return vec3(point * cosh(delta_t) + normalize(vector)*sinh(delta_t));
	}

	//3. ? 

	//4. TODO jó ez?
	static vec3 ponthozKepestPont(vec3 point, vec3 vector) {
		return vec3(point.x + vector.x, point.y + vector.y, point.z + vector.z);
	}

	//5. TODO jó ez?
	static vec3 rotateVector(vec3 vector, float angle) {
		return vec3(vector * cos(angle) + iranyraMeroleges(vector)*sin(angle));
	}

	//6. ?


};

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
		glGenVertexArrays(1, &vao); 
		glBindVertexArray(vao);
		glGenBuffers(1, &vbo); 		
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
	}
	void DrawGPU(int type, std::vector<vec2> vertices, vec3 color) {
		setUniform(color, "color");
		glBindVertexArray(vao);
		glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(vec2), &vertices[0], GL_STATIC_DRAW);
		glDrawArrays(type, 0, vertices.size());
	}
	~Renderer() {
		glDeleteBuffers(1, &vbo);
		glDeleteVertexArrays(1, &vao);
	}
};

Renderer* renderer;
std::vector<vec2> circlePoints;
const int nOfCirclePoints = 30;
long t;

class Hami {
protected:
	std::vector<vec3> nyal;
	vec3 pos, vec;
	vec2 proj;
	//float r; //nem kell, a pontok vetítése megoldja
	std::vector<vec2> circlePoints;
public:
	Hami(vec3 start) {
		for (int i = 0; i < nOfCirclePoints; i++) {
			float phi = i * 2.0f * M_PI / nOfCirclePoints;
			this->circlePoints.push_back(vec2(start.x+cosf(phi)/5, start.y+sinf(phi)/5)); //TODO: sugár változzon
		}
	}

	virtual void move(){}
	virtual void draw(){}

	void rotate(float phi = 0.1f) {

	}

	void animateMouth() {

	}
};

class PirosHami : public Hami {
public:
	PirosHami(vec2 start = vec2(0, 0)) : Hami(start) {}
	void draw() {
		renderer->DrawGPU(GL_TRIANGLE_FAN, this->circlePoints, vec3(1, 0, 0));
	}
	void move(){}
};

class ZoldHami : public Hami {
public:
	ZoldHami(vec2 start = vec2(0, 0)) : Hami(start){}
	void draw() {
		renderer->DrawGPU(GL_TRIANGLE_FAN, this->circlePoints, vec3(0, 1, 0));
	}
	void move() {}
};

PirosHami pirosHami;
ZoldHami zoldHami;

void onInitialization() {
	renderer = new Renderer();
	for (int i = 0; i < nOfCirclePoints; i++) {
		float phi = i * 2.0f * M_PI / nOfCirclePoints;
		circlePoints.push_back(vec2(cosf(phi), sinf(phi)));
	}
	pirosHami = PirosHami();
	zoldHami = ZoldHami(vec2(0.3f, 0.2f));
}

void onDisplay() {
	glClearColor(0, 0, 0, 0);
	glClear(GL_COLOR_BUFFER_BIT);
	renderer->DrawGPU(GL_TRIANGLE_FAN, circlePoints, vec3(0.5f,0.5f,0.5f));
	pirosHami.draw();
	zoldHami.draw();
	glutSwapBuffers();
}

void onKeyboard(unsigned char key, int pX, int pY) {
	switch (key) {
	case 'e':
		printf("Pressed key: e, going towards.\n");
		break;
	case 's':
		printf("Pressed key: s, rotating to left.\n");
		break;
	case 'f':
		printf("Pressed key: f, rotating to right.\n");
		break;
	}
}

void onIdle() {
	t = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;
}

void onKeyboardUp(unsigned char key, int pX, int pY) {}
void onMouseMotion(int pX, int pY) {}
void onMouse(int button, int state, int pX, int pY) {}