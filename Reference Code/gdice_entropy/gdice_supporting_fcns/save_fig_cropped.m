function save_fig_cropped(fHandle, filename, should_rasterize)
    set(fHandle,'Units','Inches');
    pos = get(fHandle,'Position');
    set(fHandle,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
    if should_rasterize
        print(fHandle,filename,'-dpdf','-r400')
    else
        print(fHandle,filename,'-dpdf','-r0')
    end
end