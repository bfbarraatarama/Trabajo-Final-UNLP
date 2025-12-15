% SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
% =======================================================================================
% Proyecto: Trabajo Final - Método de Paneles Bidimensional Multielemento No estacionario
% Archivo: src/MP2D.py
% Autor: Bruno Francisco Barra Atarama
% Institución: 
%   Departamento de Ingeniería Aeroespacial
%   Facultad de Ingeniería
%   Universidad Nacional de La Plata
% Año: 2025
% Licencia: PolyForm Noncommercial License 1.0.0
% =======================================================================================

% Complemento de los ejemplos 'ejemploBase.ipynb' y 'ejemploCargaYGuardado.ipynb' en Matlab

clear;close all;clc;set(groot,'defaultAxesXMinorGrid','on','defaultAxesXMinorGridMode','manual');set(groot,'defaultAxesYMinorGrid','on','defaultAxesYMinorGridMode','manual');

res = load("rec/ejemploBase.mat");   % Carga de los datos guardados con MP2D.guardar_resultados(...)

% Acceso a variables de las polares
alfa = res.t;
CxTot = cellfun(@(v) v(1), res.CxyTotalRec);
CyTot = cellfun(@(v) v(2), res.CxyTotalRec);
CmTot = res.CmTotalRec;

% Graficos de las polares
subplot(1,4,1);
plot(alfa, CxTot, 'o-')
xlabel('AOA [°]')
ylabel('Cd')

subplot(1,4,2);
plot(alfa, CyTot, 'o-')
xlabel('AOA [°]')
ylabel('Cl')

subplot(1,4,3);
plot(alfa, CmTot, 'o-')
xlabel('AOA [°]')
ylabel('Cm')

% Gráfico de las distribuciones de Cp para diferentes ángulos de ataque
subplot(1,4,4); hold on
labels = {};
for it=1:3:length(res.t)
    x = res.PC_XYRec{it}{1}(1,:);
    Cp = res.CpRec{it}{1};
    plot(x, Cp)
    labels{end+1} = sprintf('%.2f °', res.t(it));
end
set(gca,'YDir','reverse')
axis padded
xlabel('x')
ylabel('Cp')
legend(labels)

% Ejemplo de gráfico de panelado de dos sólidos (ejemploCargaYGuardado.ipynb)
res = load("rec/ejemploCargaYGuardado.mat");
figure; hold on
for iS=1:2
    x = res.R_XYRec{end}{iS}(1,:);
    y = res.R_XYRec{end}{iS}(2,:);
    plot(x, y, '-o')
end
axis equal; axis padded
legend('Ala', 'Flap')
xlabel('x')
ylabel('y')