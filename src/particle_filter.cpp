/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 100;
	particles.resize(num_particles);

	double std_x = std[0]; 
	double std_y = std[1];
	double std_theta = std[2];
	// init random engine
	default_random_engine gen;
	// create independed distribution for each variable 
	normal_distribution<double> dist_x(x, std_x);
    normal_distribution<double> dist_y(y, std_y);
    normal_distribution<double> dist_theta(theta, std_theta);
	// init particles by sampling from distributions
	for (int i = 0; i < num_particles; ++i) {
		Particle p; 
		p.id = i;
		p.x =  dist_x(gen);
		p.y =  dist_y(gen);
		p.theta =  dist_theta(gen);
		p.weight = 1.0;
		particles[i] = p;
	}
	// Filter initialized from now
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	
	// init random engine
	default_random_engine gen;
	// zero check for yaw_rate
	if (fabs(yaw_rate)>0.001){
		double vel_to_yaw_rate = velocity/yaw_rate;
		double yaw_change = yaw_rate*delta_t;
		// update each particle
		for (int i; i<num_particles; ++i){
			// compute prediction according to bicycle model
			double theta = particles[i].theta + yaw_change;
			double x = particles[i].x + vel_to_yaw_rate 
				* (sin(theta) - sin(particles[i].theta));
			double y = particles[i].y + vel_to_yaw_rate 
				* (cos(particles[i].theta) - cos(theta));
			
			// add noise
			// NOTE: std_pos are the same sigmas as for GPS, and this is strange
			// i would like to use STDs of velocity and yaw_rate to sample values
			// for model computations above, but i used std_pos as provided.
			normal_distribution<double> dist_x(x, std_pos[0]);
		    normal_distribution<double> dist_y(y, std_pos[1]);
		    normal_distribution<double> dist_theta(theta, std_pos[2]);

			particles[i].x = dist_x(gen);	
			particles[i].y = dist_y(gen);	
			particles[i].theta = dist_theta(gen);
			
		}
	} else { // yaw_rate close to zero
		// update each particle
		for (int i; i<num_particles; ++i){
			double vdt = velocity*delta_t;
			double theta = particles[i].theta;
			double x = particles[i].x + vdt * cos(theta);
			double y = particles[i].y + vdt * sin(theta);
			
			// add noise (see note for nonzero case)
			normal_distribution<double> dist_x(x, std_pos[0]);
		    normal_distribution<double> dist_y(y, std_pos[1]);

			particles[i].x = dist_x(gen);	
			particles[i].y = dist_y(gen);
		}
	}	
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	
	// for each observation
	for (int i=0; i < observations.size(); ++i){
		LandmarkObs *obs = &observations[i];
		double min_dist = -1.;
		// find closest landmark
		for (int j=0; j < predicted.size(); ++j){
			double d = dist(obs->x, obs->y, predicted[j].x, predicted[j].y);
			if ((d < min_dist) || (min_dist < 0.)){
				min_dist = d;
				// assign landmark INDEX (not id)
				obs->id = j;
			}
		}
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	int obsize = observations.size();
	// for each particle
	for (int i=0; i<num_particles; ++i){
		Particle p = particles[i];
		double sin_theta = sin(p.theta);
		double cos_theta = cos(p.theta);
		// transform observations to map coordinates
		std::vector<LandmarkObs> p_observations(obsize);
		for (int j=0; j<obsize; ++j){
			p_observations[j].id = -1;
			p_observations[j].x = p.x 
				+ observations[j].x * cos_theta - observations[j].y * sin_theta;
			p_observations[j].y = p.y 
				+ observations[j].x * sin_theta + observations[j].y * cos_theta;
		}

		// select landmarks withing sensor_range
		std::vector<LandmarkObs> p_predicted;
		for (int lmp=0; lmp<map_landmarks.landmark_list.size(); ++lmp){
			Map::single_landmark_s lm = map_landmarks.landmark_list[lmp];
			double d = dist(p.x, p.y, lm.x_f, lm.y_f);
			if (d <= sensor_range){
				LandmarkObs lm_predicted;
				lm_predicted.id = lm.id_i;
				lm_predicted.x = lm.x_f;
				lm_predicted.y = lm.y_f;
				p_predicted.push_back(lm_predicted);
			}
		}

		// associate closest landmark index to each observation id  
		dataAssociation(p_predicted, p_observations);
		
		// prepare visualization data
		std::vector<int> associations;
		std::vector<double> sense_x;
		std::vector<double> sense_y;		
		for (int j=0; j<obsize; ++j){
			int plm_index = p_observations[j].id;
			if (plm_index > -1){
				associations.push_back(p_predicted[plm_index].id);
				sense_x.push_back(p_observations[j].x);
				sense_y.push_back(p_observations[j].y);
			}
		}
		// set visualization data to particle
		p = SetAssociations(p, associations, sense_x, sense_y);

		// update weight of particle
		// initialize to 1.
		p.weight = 1.0;
		double std_x = std_landmark[0];
		double std_y = std_landmark[1];
		double denom = 2 * M_PI * std_x * std_y;
		double denom_x = 2 * std_x * std_x;
		double denom_y = 2 * std_y * std_y;
		for (int j =0; j<obsize; ++j){
			int plm_index = p_observations[j].id;
			double x; 
			double y;
    		if (plm_index > -1) {
				LandmarkObs plm = p_predicted[plm_index];
				x = p_predicted[plm_index].x; 
				y = p_predicted[plm_index].y; 
			} else { // if no associated landmark found
				double norm = dist(0, 0, p_observations[j].x, p_observations[j].y);
				double scale = sensor_range;
				if (fabs(norm)>0.001){
					scale = scale / norm;
				}
				double range_x = p_observations[j].x * scale;
				double range_y = p_observations[j].y * scale;
				x = p.x + range_x * cos_theta - range_y * sin_theta;
				y = p.y + range_x * sin_theta + range_y * cos_theta;
			}
			double dev_x = p_observations[j].x - x; 
			double dev_y = p_observations[j].y - y; 
			// multiply by value from multivariate Gaussian
			p.weight *= exp(-(dev_x*dev_x/denom_x) - (dev_y*dev_y/denom_y))/denom;
		}
		// update particle
		particles[i] = p;
	}

}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	default_random_engine gen;

	// NOTE: There is no need for normalize weights with discrete_distribution (see docs) 
	// collect current weights
	weights.resize(num_particles, 0.);
	for (int i=0; i<num_particles; ++i){
		weights[i] = particles[i].weight;
		weights_normalizer += particles[i].weight;
	}
	// create new vector
	std::vector<Particle> new_particles(num_particles);
	// initialize descrete distribution
	discrete_distribution<int> disc_dist(weights.begin(), weights.end());
	for(int i=0; i<num_particles; ++i) {
        // sample index from descrete distribution
        int p_index = disc_dist(gen);
        // copy particle to new vector
        new_particles[i] = particles[p_index];
    }
    // update particles with resampled vector
    particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
