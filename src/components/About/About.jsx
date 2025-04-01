import React, { useEffect, useRef } from "react";
import { useTheme } from "../../ThemeContext";

export default function About() {
  const { darkMode } = useTheme();
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    let particles = [];
    const particleCount = window.innerWidth < 768 ? 30 : 80;

    // Set canvas size
    const resizeCanvas = () => {
      canvas.width = canvas.offsetWidth;
      canvas.height = canvas.offsetHeight;
    };

    // Initialize particles
    const initParticles = () => {
      particles = [];
      for (let i = 0; i < particleCount; i++) {
        particles.push({
          x: Math.random() * canvas.width,
          y: Math.random() * canvas.height,
          size: Math.random() * 3 + 1,
          speedX: Math.random() * 1 - 0.5,
          speedY: Math.random() * 1 - 0.5,
          color: darkMode
            ? `rgba(192, 132, 252, ${Math.random() * 0.5 + 0.1})`
            : `rgba(249, 115, 22, ${Math.random() * 0.5 + 0.1})`,
        });
      }
    };

    // Mouse interaction
    let mouseX = null;
    let mouseY = null;

    const handleMouseMove = (event) => {
      const rect = canvas.getBoundingClientRect();
      mouseX = event.clientX - rect.left;
      mouseY = event.clientY - rect.top;
    };

    const handleMouseLeave = () => {
      mouseX = null;
      mouseY = null;
    };

    // Animation loop
    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Update particles
      for (let i = 0; i < particles.length; i++) {
        const p = particles[i];

        // Mouse attraction
        if (mouseX !== null && mouseY !== null) {
          const dx = mouseX - p.x;
          const dy = mouseY - p.y;
          const distance = Math.sqrt(dx * dx + dy * dy);

          if (distance < 100) {
            const forceDirectionX = dx / distance;
            const forceDirectionY = dy / distance;
            const force = (100 - distance) / 100;

            p.x -= forceDirectionX * force * 2;
            p.y -= forceDirectionY * force * 2;
          }
        }

        // Movement
        p.x += p.speedX;
        p.y += p.speedY;

        // Bounce off edges
        if (p.x < 0 || p.x > canvas.width) p.speedX *= -1;
        if (p.y < 0 || p.y > canvas.height) p.speedY *= -1;

        // Draw particle
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
        ctx.fillStyle = p.color;
        ctx.fill();

        // Draw connections
        for (let j = i + 1; j < particles.length; j++) {
          const p2 = particles[j];
          const dx = p.x - p2.x;
          const dy = p.y - p2.y;
          const distance = Math.sqrt(dx * dx + dy * dy);

          if (distance < 100) {
            ctx.beginPath();
            ctx.strokeStyle = darkMode
              ? `rgba(192, 132, 252, ${1 - distance / 100})`
              : `rgba(249, 115, 22, ${1 - distance / 100})`;
            ctx.lineWidth = 0.5;
            ctx.moveTo(p.x, p.y);
            ctx.lineTo(p2.x, p2.y);
            ctx.stroke();
          }
        }
      }

      requestAnimationFrame(animate);
    };

    // Setup
    resizeCanvas();
    initParticles();
    canvas.addEventListener("mousemove", handleMouseMove);
    canvas.addEventListener("mouseleave", handleMouseLeave);
    const animationId = requestAnimationFrame(animate);

    // Cleanup
    return () => {
      canvas.removeEventListener("mousemove", handleMouseMove);
      canvas.removeEventListener("mouseleave", handleMouseLeave);
      cancelAnimationFrame(animationId);
    };
  }, [darkMode]);

  return (
    <div
      className={`relative overflow-hidden py-20 ${
        darkMode ? "bg-gray-900" : "bg-gray-50"
      }`}
    >
      {/* Interactive Canvas Background */}
      <canvas
        ref={canvasRef}
        className="absolute top-0 left-0 w-full h-full pointer-events-none"
      />

      {/* Content (same as before) */}
      <div className="relative z-10">
        <div className="container mx-auto px-8 md:px-12 xl:px-16">
          <div className="md:flex md:gap-12 lg:items-center">
            {/* Image Section */}
            <div className="image-container md:w-5/12 lg:w-4/12 transform transition duration-500 hover:scale-105">
              <img
                src={"./aac2.jpg"}
                alt="All About Coding"
                className={`image-flip rounded-3xl shadow-2xl border-4 ${
                  darkMode ? "border-purple-500" : "border-orange-500"
                }`}
                loading="lazy"
              />
            </div>

            {/* Content Section */}
            <div className="mt-10 md:mt-0 md:w-7/12 lg:w-8/12">
              {/* Heading */}
              <h2
                className={`text-4xl font-extrabold md:text-5xl lg:text-6xl bg-clip-text text-transparent bg-gradient-to-r ${
                  darkMode
                    ? "from-purple-400 to-pink-500"
                    : "from-orange-500 to-purple-600"
                }`}
              >
                Welcome to All About Coding
              </h2>

              {/* Description */}
              <p
                className={`mt-6 text-lg leading-relaxed ${
                  darkMode ? "text-gray-300" : "text-gray-800"
                }`}
              >
                At All About Coding, we are committed to helping you master Data
                Structures and Algorithms (DSA) while building a strong
                foundation for a successful tech career. Our mission is to
                empower learners with expertly curated resources and actionable
                insights tailored for success in academics, internships, and
                placements.
              </p>

              {/* What We Offer Section */}
              <h3
                className={`mt-8 text-2xl font-semibold bg-clip-text text-transparent bg-gradient-to-r ${
                  darkMode
                    ? "from-purple-400 to-pink-500"
                    : "from-orange-500 to-purple-600"
                }`}
              >
                What We Offer:
              </h3>
              <ul className="mt-6 space-y-6">
                {[
                  "Comprehensive DSA Resources: Simplified explanations with problem sets.",
                  "Placement-Focused Roadmaps: Yearly guides to enhance skills and projects.",
                  "Expert Insights: Strategies and tips for coding interviews.",
                  "Community Support: Engage with like-minded learners and professionals.",
                ].map((item, index) => (
                  <li
                    key={index}
                    className={`flex items-start transition duration-300 hover:translate-x-2 ${
                      darkMode ? "text-gray-300" : "text-gray-700"
                    }`}
                  >
                    <span
                      className={`flex w-6 h-6 mt-1 mr-4 rounded-full items-center justify-center text-white font-bold ${
                        darkMode ? "bg-purple-500" : "bg-orange-500"
                      }`}
                    >
                      ✓
                    </span>
                    <span>{item}</span>
                  </li>
                ))}
              </ul>

              {/* Closing Paragraph */}
              <p
                className={`mt-8 text-lg leading-relaxed ${
                  darkMode ? "text-gray-300" : "text-gray-800"
                }`}
              >
                Whether you're just beginning or honing your expertise, All
                About Coding ensures an engaging and effective learning
                experience. Join us and unlock your full potential!
              </p>

              {/* Call-to-Action Button */}
              <div className="mt-8 flex justify-center">
                <a
                  href="https://chat.whatsapp.com/INdHcEdh3ieGE5eHDg4vea"
                  target="_blank"
                  rel="noopener noreferrer"
                  className={`animated-gradient inline-block px-8 py-4 text-white font-semibold rounded-xl shadow-lg hover:shadow-2xl transform transition duration-300 hover:scale-110 ${
                    darkMode
                      ? "bg-gradient-to-r from-purple-600 to-pink-600"
                      : "bg-gradient-to-r from-orange-500 to-purple-600"
                  }`}
                  aria-label="Join Our WhatsApp Community"
                >
                  Join Our Community
                </a>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
