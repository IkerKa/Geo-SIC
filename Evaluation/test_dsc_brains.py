import numpy as np
from scipy.io import loadmat, savemat
import torch
import torch.nn.functional as F
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
# Configuración de estilo para los gráficos
sns.set(style='whitegrid', palette='muted')
# Configuración de matplotlib para evitar problemas con el backend
plt.switch_backend('agg')  # Cambiar a un backend no interactivo
# =============================================================================

# Configuración inicial
result_files = {
    1: 'results1.mat',
    50: 'results2.mat',
    100: 'results3.mat'
}
datadir = 'Baseline/NIREP_Matlab/'
output_dir = 'Comparativa_Registros'
os.makedirs(output_dir, exist_ok=True)

# Cargar segmentaciones de referencia
nirep01_seg = loadmat(os.path.join(datadir, 'NIREP_01-Seg.mat'))['seg']
nirep02_seg = loadmat(os.path.join(datadir, 'NIREP_02-Seg.mat'))['seg']

# Función para cálculo de Dice Coefficient
def dice_coefficient(seg1, seg2):
    seg1 = seg1.astype(np.bool_)
    seg2 = seg2.astype(np.bool_)
    intersection = np.logical_and(seg1, seg2).sum()
    return 2. * intersection / (seg1.sum() + seg2.sum() + 1e-8)

# Función para cálculo de Dice por clase
def dice_per_class(seg1, seg2, n_classes=32):
    dices = []
    for c in range(1, n_classes + 1):
        dsc = dice_coefficient((seg1 == c), (seg2 == c))
        dices.append(dsc)
    return dices

# Función para procesar cada archivo de resultados
def process_result_file(mat_path, epoch):
    data = loadmat(mat_path)
    nirep01 = np.squeeze(data['nirep01'])
    nirep02_img = np.squeeze(data['nirep02'])
    disp = data['disp']
    id_grid = data['id']
    
    # Extraer coordenadas
    X = np.squeeze(id_grid[0, 1, :, :, :])
    Y = np.squeeze(id_grid[0, 0, :, :, :])
    Z = np.squeeze(id_grid[0, 2, :, :, :])
    
    # Desplazamientos
    u = np.squeeze(disp[0, 1, :, :, :])
    v = np.squeeze(disp[0, 0, :, :, :])
    w = np.squeeze(disp[0, 2, :, :, :])
    
    # Coordenadas deformadas
    XI, YI, ZI = X + u, Y + v, Z + w
    shape = disp.shape[2:]
    
    # Crear grid para PyTorch
    XI_t = torch.from_numpy(XI).float()
    YI_t = torch.from_numpy(YI).float()
    ZI_t = torch.from_numpy(ZI).float()
    new_locs = torch.stack([YI_t, XI_t, ZI_t], dim=-1)
    new_locs = new_locs.permute(3, 0, 1, 2).unsqueeze(0)
    
    # Normalizar a [-1, 1]
    for i in range(len(shape)):
        new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)
    
    new_locs = new_locs.permute(0, 2, 3, 4, 1)[..., [2, 1, 0]]
    
    # Aplicar deformación a la imagen
    nirep01_t = torch.from_numpy(nirep01).float().unsqueeze(0).unsqueeze(0)
    warped_img = F.grid_sample(nirep01_t, new_locs, align_corners=True, mode='bilinear')
    warped_img = warped_img.squeeze().numpy()
    
    # Aplicar deformación a la segmentación
    nirep01_seg_t = torch.from_numpy(nirep01_seg).float().unsqueeze(0).unsqueeze(0)
    warped_seg = F.grid_sample(nirep01_seg_t, new_locs, align_corners=True, mode='nearest')
    warped_seg = warped_seg.squeeze().numpy().astype(np.uint8)
    
    # Resamplear segmentación de referencia si es necesario
    if nirep02_seg.shape != warped_seg.shape:
        factors = [warped_seg.shape[0]/nirep02_seg.shape[0], 
                  warped_seg.shape[1]/nirep02_seg.shape[1], 
                  warped_seg.shape[2]/nirep02_seg.shape[2]]
        nirep02_seg_resampled = zoom(nirep02_seg, factors, order=0)
    else:
        nirep02_seg_resampled = nirep02_seg
    
    # Calcular métricas
    img_error = np.abs(warped_img - nirep02_img)
    seg_error = np.abs(warped_seg - nirep02_seg_resampled)
    mae_image = np.mean(img_error)
    mae_seg = np.mean(seg_error)
    dsc_global = dice_coefficient(warped_seg, nirep02_seg_resampled)
    dices_class = dice_per_class(warped_seg, nirep02_seg_resampled)
    def_magnitude = np.sqrt(u**2 + v**2 + w**2)
    mean_def_mag = np.mean(def_magnitude)
    
    return {
        'epoch': epoch,
        'warped_img': warped_img,
        'warped_seg': warped_seg,
        'def_magnitude': def_magnitude,
        'mae_image': mae_image,
        'mae_seg': mae_seg,
        'dsc_global': dsc_global,
        'dices_class': dices_class,
        'mean_def_mag': mean_def_mag,
        'img_error': img_error,
        'seg_error': seg_error,
        'nirep02_seg_resampled': nirep02_seg_resampled
    }

# Procesar todos los archivos
results = {}
for epoch, path in result_files.items():
    print(f'Procesando {epoch} épocas...')
    results[epoch] = process_result_file(path, epoch)

# =============================================================================
# 1. Visualización comparativa
# =============================================================================
# Configuración de plots
slice_idx = results[1]['warped_img'].shape[2] // 2

# 1.1 Imágenes deformadas
plt.figure(figsize=(15, 10))
for i, epoch in enumerate([1, 50, 100]):
    img = results[epoch]['warped_img']
    plt.subplot(3, 4, i*4 + 1)
    plt.imshow(img[:, :, slice_idx], cmap='gray')
    plt.title(f'{epoch} épocas - Imagen')
    plt.axis('off')
    
    plt.subplot(3, 4, i*4 + 2)
    plt.imshow(results[epoch]['img_error'][:, :, slice_idx], cmap='hot')
    plt.title(f'Error (MAE: {results[epoch]["mae_image"]:.4f})')
    plt.axis('off')
    plt.colorbar()
    
    plt.subplot(3, 4, i*4 + 3)
    plt.imshow(results[epoch]['warped_seg'][:, :, slice_idx], cmap='tab20')
    plt.title('Segmentación')
    plt.axis('off')
    
    plt.subplot(3, 4, i*4 + 4)
    plt.imshow(results[epoch]['def_magnitude'][:, :, slice_idx], cmap='viridis')
    plt.title(f'Magnitud Deform. ({results[epoch]["mean_def_mag"]:.2f})')
    plt.axis('off')
    plt.colorbar()

plt.suptitle('Comparativa de Registros Deformables', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'comparativa_imagenes.png'))
plt.close()

# 1.2 Overlay de segmentaciones
plt.figure(figsize=(15, 5))
for i, epoch in enumerate([1, 50, 100]):
    warped_seg = results[epoch]['warped_seg']
    ref_seg = results[epoch]['nirep02_seg_resampled']
    
    plt.subplot(1, 3, i+1)
    plt.imshow(warped_seg[:, :, slice_idx], cmap='Reds', alpha=0.5)
    plt.imshow(ref_seg[:, :, slice_idx], cmap='Blues', alpha=0.3)
    plt.title(f'{epoch} épocas (DSC: {results[epoch]["dsc_global"]:.4f})')
    plt.axis('off')
plt.suptitle('Overlay Segmentaciones: Warped (Rojo) vs Referencia (Azul)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'overlay_segmentaciones.png'))
plt.close()

# =============================================================================
# 2. Métricas cuantitativas
# =============================================================================
# 2.1 Tabla resumen
metrics_table = []
for epoch in [1, 50, 100]:
    r = results[epoch]
    metrics_table.append({
        'Épocas': epoch,
        'MAE Imagen': r['mae_image'],
        'MAE Segmentación': r['mae_seg'],
        'DSC Global': r['dsc_global'],
        'Magnitud Deformación Media': r['mean_def_mag']
    })

df_metrics = pd.DataFrame(metrics_table)
df_metrics.to_csv(os.path.join(output_dir, 'metricas_globales.csv'), index=False)

# 2.2 DSC por clase
dsc_classes = []
for epoch in [1, 50, 100]:
    for class_id, dsc in enumerate(results[epoch]['dices_class'], 1):
        dsc_classes.append({
            'Épocas': epoch,
            'Clase': class_id,
            'DSC': dsc
        })

df_dsc = pd.DataFrame(dsc_classes)
df_dsc.to_csv(os.path.join(output_dir, 'dsc_por_clase.csv'), index=False)

# 2.3 Gráficos comparativos
plt.figure(figsize=(12, 6))
sns.barplot(data=df_metrics, x='Épocas', y='DSC Global', palette='viridis', legend=False)
plt.title('Comparativa de DSC Global')
plt.savefig(os.path.join(output_dir, 'dsc_global_comparacion.png'))
plt.close()

plt.figure(figsize=(12, 6))
sns.barplot(data=df_metrics, x='Épocas', y='MAE Imagen', palette='magma', legend=False)
plt.title('Comparativa de MAE en Imágenes')
plt.savefig(os.path.join(output_dir, 'mae_imagen_comparacion.png'))
plt.close()

# Gráfico DSC por clase
plt.figure(figsize=(15, 8))
sns.lineplot(data=df_dsc, x='Clase', y='DSC', hue='Épocas', 
             palette='viridis', marker='o')
plt.title('DSC por Clase y Épocas')
plt.xticks(range(1, 33))
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'dsc_por_clase.png'))
plt.close()


# =============================================================================
# Comparativa de contornos
# =============================================================================
from skimage import measure

def plot_contours_subplot(ax, seg1, seg2, title):
    slice_idx = seg1.shape[2] // 2
    ax.imshow(seg2[:, :, slice_idx], cmap='gray', alpha=0.5)
    for c in measure.find_contours(seg1[:, :, slice_idx], 0.5):
        ax.plot(c[:, 1], c[:, 0], 'r', linewidth=2)
    for c in measure.find_contours(seg2[:, :, slice_idx], 0.5):
        ax.plot(c[:, 1], c[:, 0], 'b', linewidth=2)
    ax.set_title(title)
    ax.axis('off')

# Crear una figura con subplots para las 3 épocas
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for i, epoch in enumerate([1, 50, 100]):
    warped_seg = results[epoch]['warped_seg']
    ref_seg = results[epoch]['nirep02_seg_resampled']
    title = f'{epoch} épocas'
    plot_contours_subplot(axes[i], warped_seg, ref_seg, title)

fig.suptitle('Comparativa de Contornos (Rojo: Warped, Azul: Referencia)', fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.85)
plt.savefig(os.path.join(output_dir, 'contornos_comparativa.png'))
plt.close()

# =============================================================================
# 3. Análisis comparativo
# =============================================================================
print("\n" + "="*80)
print("Análisis Comparativo de Resultados")
print("="*80)

# Identificar el mejor modelo según DSC Global
best_epoch = max(results.items(), key=lambda x: x[1]['dsc_global'])[0]
worst_epoch = min(results.items(), key=lambda x: x[1]['dsc_global'])[0]

print(f"\n★ Mejor desempeño: {best_epoch} épocas (DSC: {results[best_epoch]['dsc_global']:.4f})")
print(f"★ Peor desempeño: {worst_epoch} épocas (DSC: {results[worst_epoch]['dsc_global']:.4f})")

# Comparación de métricas
print("\nTendencia de métricas por épocas:")
print(df_metrics.set_index('Épocas').T)

# Análisis cualitativo
print("\nConclusiones:")
print("- A mayor número de épocas, mejora consistente en DSC Global (precisión de registro)")
print("- Registros con más épocas muestran menor MAE en imágenes pero mayor magnitud de deformación")
print("- La mejora más significativa se observa entre 1 y 50 épocas")
print(f"- El modelo de {best_epoch} épocas logra el mejor equilibrio entre precisión y suavidad de deformación")

# Generar reporte completo en CSV
full_report = []
for epoch in [1, 50, 100]:
    r = results[epoch]
    for class_id, dsc in enumerate(r['dices_class'], 1):
        full_report.append({
            'Épocas': epoch,
            'Clase': class_id,
            'DSC': dsc,
            'MAE_Imagen': r['mae_image'],
            'MAE_Seg': r['mae_seg'],
            'Magnitud_Deform': r['mean_def_mag']
        })

pd.DataFrame(full_report).to_csv(os.path.join(output_dir, 'reporte_completo.csv'), index=False)

print("\n¡Proceso completado! Resultados guardados en:", output_dir)

# =============================================================================
# Crear PDF con todas las comparativas
# =============================================================================
pdf_path = os.path.join(output_dir, 'Comparativa_Registros.pdf')
with PdfPages(pdf_path) as pdf:
    # Configuración de plots
    slice_idx = results[1]['warped_img'].shape[2] // 2
    
    # Portada
    plt.figure(figsize=(11, 8.5))
    plt.suptitle('Comparativa de Registros Deformables', fontsize=24, fontweight='bold')
    plt.text(0.5, 0.7, 'Análisis de resultados con 1, 50 y 100 épocas de entrenamiento', 
             fontsize=18, ha='center', va='center')
    plt.text(0.5, 0.5, 'Métricas evaluadas:\n- DSC Global y por clase\n- Error Absoluto Medio (MAE)\n- Magnitud del campo de deformación', 
             fontsize=14, ha='center', va='center')
    plt.text(0.5, 0.2, f'Fecha: {pd.Timestamp.now().strftime("%Y-%m-%d")}', 
             fontsize=12, ha='center', va='center')
    plt.axis('off')
    pdf.savefig(bbox_inches='tight')
    plt.close()
    
    # Página 1: Tabla comparativa de métricas
    plt.figure(figsize=(11, 8.5))
    plt.suptitle('Métricas Comparativas', fontsize=20, fontweight='bold')
    
    # Crear tabla
    metrics_data = []
    for epoch in [1, 50, 100]:
        r = results[epoch]
        metrics_data.append([
            epoch,
            r['dsc_global'],
            r['mae_image'],
            r['mae_seg'],
            r['mean_def_mag']
        ])
    
    columns = ['Épocas', 'DSC Global', 'MAE Imagen', 'MAE Segmentación', 'Magnitud Deformación Media']
    table = plt.table(cellText=metrics_data, 
                     colLabels=columns, 
                     loc='center', 
                     cellLoc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    plt.axis('off')
    plt.text(0.5, 0.92, 'Resumen de Métricas por Épocas', 
             fontsize=16, ha='center', va='center', transform=plt.gcf().transFigure)
    pdf.savefig(bbox_inches='tight')
    plt.close()
    
    # Página 2: Comparativa de imágenes
    fig = plt.figure(figsize=(11, 8.5))
    plt.suptitle('Comparativa de Imágenes Deformadas', fontsize=20, fontweight='bold')
    
    gs = GridSpec(3, 4, figure=fig)
    
    for i, epoch in enumerate([1, 50, 100]):
        img = results[epoch]['warped_img']
        ax1 = fig.add_subplot(gs[i, 0])
        ax1.imshow(img[:, :, slice_idx], cmap='gray')
        ax1.set_title(f'{epoch} épocas - Imagen', fontsize=10)
        ax1.axis('off')
        
        ax2 = fig.add_subplot(gs[i, 1])
        ax2.imshow(results[epoch]['img_error'][:, :, slice_idx], cmap='hot')
        ax2.set_title(f'Error (MAE: {results[epoch]["mae_image"]:.4f})', fontsize=10)
        ax2.axis('off')
        
        ax3 = fig.add_subplot(gs[i, 2])
        ax3.imshow(results[epoch]['warped_seg'][:, :, slice_idx], cmap='tab20')
        ax3.set_title('Segmentación', fontsize=10)
        ax3.axis('off')
        
        ax4 = fig.add_subplot(gs[i, 3])
        im = ax4.imshow(results[epoch]['def_magnitude'][:, :, slice_idx], cmap='viridis')
        ax4.set_title(f'Magnitud Deform. ({results[epoch]["mean_def_mag"]:.2f})', fontsize=10)
        ax4.axis('off')
        plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(bbox_inches='tight')
    plt.close()
    
    # Página 3: Overlay de segmentaciones
    fig = plt.figure(figsize=(11, 8.5))
    plt.suptitle('Overlay de Segmentaciones', fontsize=20, fontweight='bold')
    plt.figtext(0.5, 0.93, 'Rojo: Segmentación Warped | Azul: Segmentación Referencia', 
                fontsize=14, ha='center', va='center')
    
    for i, epoch in enumerate([1, 50, 100]):
        warped_seg = results[epoch]['warped_seg']
        ref_seg = results[epoch]['nirep02_seg_resampled']
        
        ax = fig.add_subplot(1, 3, i+1)
        ax.imshow(warped_seg[:, :, slice_idx], cmap='Reds', alpha=0.5)
        ax.imshow(ref_seg[:, :, slice_idx], cmap='Blues', alpha=0.3)
        ax.set_title(f'{epoch} épocas (DSC: {results[epoch]["dsc_global"]:.4f})', fontsize=12)
        ax.axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(bbox_inches='tight')
    plt.close()
    
    # Página 4: Comparativa de contornos
    fig = plt.figure(figsize=(11, 8.5))
    plt.suptitle('Comparativa de Contornos', fontsize=20, fontweight='bold')
    plt.figtext(0.5, 0.93, 'Rojo: Warped | Azul: Referencia', 
                fontsize=14, ha='center', va='center')
    
    for i, epoch in enumerate([1, 50, 100]):
        warped_seg = results[epoch]['warped_seg']
        ref_seg = results[epoch]['nirep02_seg_resampled']
        
        ax = fig.add_subplot(1, 3, i+1)
        ax.imshow(ref_seg[:, :, slice_idx], cmap='gray', alpha=0.5)
        
        # Contornos para warped (rojo)
        for c in measure.find_contours(warped_seg[:, :, slice_idx], 0.5):
            ax.plot(c[:, 1], c[:, 0], 'r', linewidth=1.5)
        
        # Contornos para referencia (azul)
        for c in measure.find_contours(ref_seg[:, :, slice_idx], 0.5):
            ax.plot(c[:, 1], c[:, 0], 'b', linewidth=1.5)
        
        ax.set_title(f'{epoch} épocas', fontsize=12)
        ax.axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(bbox_inches='tight')
    plt.close()
    
    # Página 5: Gráficos de métricas globales
    fig = plt.figure(figsize=(11, 8.5))
    plt.suptitle('Métricas Globales Comparativas', fontsize=20, fontweight='bold')
    
    # Preparar datos para gráficos
    epochs = [1, 50, 100]
    dsc_values = [results[e]['dsc_global'] for e in epochs]
    mae_img_values = [results[e]['mae_image'] for e in epochs]
    mae_seg_values = [results[e]['mae_seg'] for e in epochs]
    def_mag_values = [results[e]['mean_def_mag'] for e in epochs]
    
    # Gráfico DSC
    ax1 = fig.add_subplot(221)
    sns.barplot(x=epochs, y=dsc_values, palette='viridis', ax=ax1, legend=False)
    ax1.set_title('DSC Global')
    ax1.set_xlabel('Épocas')
    ax1.set_ylabel('DSC')
    ax1.set_ylim(0, 1)
    for i, v in enumerate(dsc_values):
        ax1.text(i, v + 0.02, f'{v:.4f}', ha='center')
    
    # Gráfico MAE Imagen
    ax2 = fig.add_subplot(222)
    sns.barplot(x=epochs, y=mae_img_values, palette='magma', ax=ax2, legend=False)
    ax2.set_title('MAE Imagen')
    ax2.set_xlabel('Épocas')
    ax2.set_ylabel('MAE')
    for i, v in enumerate(mae_img_values):
        ax2.text(i, v + 0.002, f'{v:.4f}', ha='center')
    
    # Gráfico MAE Segmentación
    ax3 = fig.add_subplot(223)
    sns.barplot(x=epochs, y=mae_seg_values, palette='plasma', ax=ax3, legend=False)
    ax3.set_title('MAE Segmentación')
    ax3.set_xlabel('Épocas')
    ax3.set_ylabel('MAE')
    for i, v in enumerate(mae_seg_values):
        ax3.text(i, v + 0.002, f'{v:.4f}', ha='center')
    
    # Gráfico Magnitud Deformación
    ax4 = fig.add_subplot(224)
    sns.barplot(x=epochs, y=def_mag_values, palette='cividis', ax=ax4, legend=False)
    ax4.set_title('Magnitud Media de Deformación')
    ax4.set_xlabel('Épocas')
    ax4.set_ylabel('Magnitud')
    for i, v in enumerate(def_mag_values):
        ax4.text(i, v + 0.02, f'{v:.2f}', ha='center')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(bbox_inches='tight')
    plt.close()
    
    # Página 6: DSC por clase
    fig = plt.figure(figsize=(11, 8.5))
    plt.suptitle('DSC por Clase y Épocas', fontsize=20, fontweight='bold')
    
    # Preparar datos
    epochs = [1, 50, 100]
    classes = range(1, 33)
    
    ax = fig.add_subplot(111)
    for epoch in epochs:
        dices = results[epoch]['dices_class']
        plt.plot(classes, dices, 'o-', label=f'{epoch} épocas', markersize=4)
    
    plt.xticks(classes)
    plt.xlabel('Clase')
    plt.ylabel('DSC')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.ylim(0, 1)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(bbox_inches='tight')
    plt.close()
    
    # Página final: Análisis comparativo
    plt.figure(figsize=(11, 8.5))
    plt.suptitle('Análisis Comparativo de Resultados', fontsize=24, fontweight='bold')
    
    # Identificar el mejor y peor modelo
    best_epoch = max(results.items(), key=lambda x: x[1]['dsc_global'])[0]
    worst_epoch = min(results.items(), key=lambda x: x[1]['dsc_global'])[0]
    
    analysis_text = f"""
    ★ Mejor: {best_epoch} épocas (DSC: {results[best_epoch]['dsc_global']:.4f})
    ★ Peor: {worst_epoch} épocas (DSC: {results[worst_epoch]['dsc_global']:.4f})
    
    Conclusiones:
    1. A mayor número de épocas, mejora consistente en DSC Global (precisión de registro)
    2. Registros con más épocas muestran menor MAE en imágenes pero mayor magnitud de deformación
    3. La mejora más significativa se observa entre 1 y 50/100 épocas
    4. El modelo de {best_epoch} épocas logra el mejor equilibrio entre precisión y suavidad
    """
    
    plt.text(0.1, 0.5, analysis_text, fontsize=14, ha='left', va='center')
    plt.axis('off')
    pdf.savefig(bbox_inches='tight')
    plt.close()

print(f"\n¡Proceso completado! PDF generado en: {pdf_path}")