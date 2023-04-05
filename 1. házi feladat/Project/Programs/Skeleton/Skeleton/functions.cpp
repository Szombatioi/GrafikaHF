////A vektorok normalizálása mindig a hívó feladata! Ne adjunk vissza feleslegesen normalizált vektort

vec3 hyNormalize(vec3 vector){
    return vector / hyDot(vector, vector);
}

float hyDot(vec3 p, vec3 q){
    return p.x*q.x + p.y*q.y - p.z*q.z;
}

vec3 hyCross(){
    return vec3(); //TODO
}

// 1. Egy irányra merőleges irány állítása.
vec3 hyPerp(vec3 vector){
    return vec3(); //TODO:
}

// 2. Adott pontból és sebesség vektorral induló pont helyének és sebesség vektorának számítása t idővel később.
vec3 hyMoveP(vec3 point, vec3 vector, float t){
    return point * coshf(t) + hyNormalize(vector) * sinhf(t);
}
vec3 hyMoveV(vec3 point, vec3 vector, float t){
    return point * sinhf(t) + hyNormalize(vector) * sinhf(t);
}

// 3. Egy ponthoz képest egy másik pont irányának és távolságának meghatározása.
vec3 hyDir(vec3 p, vec3 q){
    return vec3((q - p * coshf(0.0001)) / sinhf(0.0001)); //0.0001 -> precizitás? t -> 0 ? 
}

vec3 hyDist(vec3 p, vec3 q){
    return acoshf(-hyDot(p,q));
}

// 4. Egy ponthoz képest adott irányban és távolságra lévő pont előállítása.
vec3 hyProduceP(vec3 point, vec3 vector, float distance){
    return point * coshf(dist) + hyNormalize(vector) *  sinhf(dist);
    //vector-t változtatni kellene itt?
}

// 5. Egy pontban egy vektor elforgatása adott szöggel.
vec3 hyRotate(vec3 vector, float phi){
    vec3 v = hyNormalize(vector);
    return vec3(v * cosf(phi) + hyPerp(v) * sinf(phi));
}

// 6. Egy közelítő pont és sebességvektorhoz a geometria szabályait teljesítő, közeli pont és sebesség választása.
////Ha mozog a pont, akkor mindig vissza kell dobnunk a síkra
vec3 hyNearP(vec3 point){
    point.z = sqrt(point.x * point.x + point.y * point.y + 1); ///Ehelyett centrális visszavetítés kéne
	return point;
}

vec3 hyNearV(vec3 point, vec3 vector){
    float lambda = hyDot(point, vector);
	return vector + lambda * point;
}